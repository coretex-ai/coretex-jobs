args <- commandArgs(trailingOnly = TRUE)
library(reticulate)

# Import python modules
builtins <- import_builtins()
zipfile <- import("zipfile")
ctx <- import("coretex")
ctx_folder_manager <- import("coretex.folder_manager")

# Contents
# 2. Load packages
# 3. Start DADA2 pipeline
# Step3.1: Set up folder with input files (after primer removal in cutadapt)
# Step3.2: Match file names
# Step3.3: Read quality assessment
# Step3.4: Filter and trim
# Step3.5: DADA denoising
# 3.5.3 Merge reads
# Step3.6: ASV table construction
# Step3.7: Track reads
# Step3.8: Taxonomy assignment using DADA2 or DECIPHER
# Step3.9: Process files
# 4. MSA
# 5. Phylogenetic tree

########## 2.Load packages ##########
print("Step 2: Loading packages")

installPhyloseqExtended <- function(attempt = NULL) {
    tryCatch(
        {
            if (is.null(attempt)) attempt <- 1
            if (attempt > 3) stop("Failed to install phyloseq.extended package")

            # I have no idea how to install this using conda
            if (!requireNamespace("phyloseq.extended", quietly = TRUE)) {
                print("Installing \"phyloseq.extended\" package")
                remotes::install_github(
                    "mahendra-mariadassou/phyloseq-extended",
                    ref = "dev",
                    quiet = TRUE
                )
            }
        },
        error = function(e) {
            print("Retrying install of phyloseq.extended package")
            installPhyloseqExtended(attempt + 1)
        }
    )
}

installPhyloseqExtended()

library("devtools")
library(dada2)
library("Biostrings")
library(vegan)
library(ggplot2)
library(data.table)
library(phyloseq)
library(readr)
library(phyloseq.extended)
library(DECIPHER)

########### 3.Start DADA2 pipeline #############
# Start with cutadapt and read quality assessment in fastqc/multiqc (refer to README file)
# Step3.1: Set up folder with input files (after primer removal in cutadapt)
# Step3.2: Match file names
# Step3.3: Read quality assessment
# Step3.4: Filter and trim
# Step3.5: DADA denoising-
# 3.5.1 Learn error rates and plot them
# 3.5.2 Sample inference
# 3.5.3 Merge reads
# Step3.6: ASV table construction
# 6.1 Remove chimeras
# Step3.7: Track reads
# Step3.8: Taxonomy assignment using DADA2 assignment (SILVA Db)
# 8.1 Save the Seqtab.RData as a fasta file with ASVs
# 8.2 Save taxonomy table
# Step9: MSA with DECIPHER
# Step10: Phylogenetic tree with Fasttree
# Step11: Create the phyloseq object

# Step3.1: Input files - start with files after removing primers

forward_pattern <- "L001_R1_001.fastq.gz"
reverse_pattern <- "L001_R2_001.fastq.gz"

getSampleIdColumnName <- function(metadata) {
    caseInsensitiveColumnNames <- c(
        "id", "sampleid", "sample id", "sample-id", "sample_id", "sample.id", "featureid", "feature id", "feature-id"
    )
    # Make sure caseInsensitiveColumnNames is always lowercase
    lapply(caseInsensitiveColumnNames, tolower)

    caseSensitiveColumnNames <- c(
        "#SampleID", "#Sample ID", "#OTUID", "#OTU ID", "sample_name"
    )

    metadataColumns = colnames(metadata)
    lowercaseColumns = lapply(metadataColumns, tolower)

    for (caseInsensitiveColumnName in caseInsensitiveColumnNames) {
        result <- which(lowercaseColumns == caseInsensitiveColumnName)
        if (length(result) > 0) {
            return(metadataColumns[result])
        }
    }

    for (caseSensitiveColumnName in caseInsensitiveColumnNames) {
        if (caseSensitiveColumnName %in% metadataColumns) {
            return(caseSensitiveColumnName)
        }
    }

    stop(paste(
        "Failed to determine which column contains sample IDs/names, available names are (case insensitive):",
        paste(c(caseInsensitiveColumnNames), collapse = ", "),
        ", and (case sensitive)",
        paste(c(caseSensitiveColumnNames), collapse = ", ")
    ))
}

loadMetadata <- function(metadataSample) {
    metadata_csv_path <- builtins$str(
        metadataSample$joinPath("metadata.csv")
    )

    if (file.exists(metadata_csv_path)) {
        # Default SampleSheet.csv format
        metadata <- read.table(
            metadata_csv_path,
            sep = ",",
            header = TRUE,
            check.names = TRUE
        )
    } else {
        # Format accepted by qiime2
        metadata_tsv_path <- builtins$str(
            metadataSample$joinPath("metadata.tsv")
        )

        if (!file.exists(metadata_tsv_path)) {
            stop("Metadata file not found")
        }

        metadata <- read.table(
            metadata_tsv_path,
            sep = "\t",
            header = TRUE,
            check.names = TRUE
        )

        # qiime has 1 extra row after header which contains types
        metadata <- metadata[-1,]
    }

    # Remove leading and trailing whitespace
    colnames(metadata) <- lapply(colnames(metadata), trimws)

    stringColumns <- names(metadata)[vapply(metadata, is.character, logical(1))]
    metadata[, stringColumns] <- lapply(metadata[, stringColumns], trimws)

    sampleIdColumn <- getSampleIdColumnName(metadata)
    print(paste("Matched metadata sample ID/name column to", sampleIdColumn))

    print("Renaming metadata sample ID/name column to \"sampleId\"")
    names(metadata)[names(metadata) == sampleIdColumn] <- "sampleId"

    print("Metadata")
    print(colnames(metadata))
    print(head(metadata))

    print(metadata$sampleId)

    # assign the names of samples (01Sat1...) to metadata rows instead of 1,2,3...
    row.names(metadata) <- metadata$sampleId
    metadata$sampleId <- as.factor(metadata$sampleId)

    return(metadata)
}

trackPctColumns <- function(names, track_matrix) {
    for (name in names) {
        percentage = cbind(round(track_matrix[, name] / track_matrix[, "input"] * 100, digits = 2))

        columnName <- paste0(name, "Pct")
        colnames(percentage) <- c(columnName)

        track_matrix <- cbind(
            track_matrix,
            percentage
        )
    }

    return(track_matrix)
}

countReads <- function(file_path) {
    file_connection <- file(file_path, "r")
    lines <- readLines(file_connection)
    close(file_connection)

    return (length(lines) %/% 4)
}

getFilteringResults <- function(read_paths, filtered_read_paths) {
    reads_in <- c()
    reads_out <- c()

    for (read_path in read_paths) {
        reads_in <- c(reads_in, countReads(read_path))
    }
    for (filtered_read_path in filtered_read_paths) {
        reads_out <- c(reads_out, countReads(filtered_read_path))
    }
    results <- data.frame("reads.in" = reads_in, "reads.out" = reads_out)
    rownames(results) <- basename(read_paths)
    return (results)
}

tryFilterAndTrim <- function(
    forward_read_paths,
    filtered_forward_read_paths,
    reverse_read_paths,
    filtered_reverse_read_paths,
    trimLeft,
    trimRight,
    truncLen,
    maxN,
    maxEE,
    truncQ,
    rm_phix = TRUE,
    compress = TRUE,
    multithread = TRUE,
    matchIDs = FALSE
) {
    tryCatch(
        expr = {
            filtering_results <- filterAndTrim(
                forward_read_paths,
                filtered_forward_read_paths,
                reverse_read_paths,
                filtered_reverse_read_paths,
                trimLeft = trimLeft,
                trimRight = trimRight,
                truncLen = truncLen,
                maxN = maxN,
                maxEE = maxEE,
                truncQ = truncQ,
                rm.phix = rm_phix,
                compress = compress,
                multithread = multithread,
                matchIDs = matchIDs
            )
        },
        error = function(cond) {
            print("Failed filterAndTrim. Retrying for unsuccessful samples")

            lo_forward_read_paths <- c()
            lo_reverse_read_paths <- c()
            lo_filtered_forward_read_paths <- c()
            lo_filtered_reverse_read_paths <- c()

            for (i in 1:length(filtered_forward_read_paths)){
                if (!file.exists(filtered_forward_read_paths[i])){
                    lo_forward_read_paths <- c(lo_forward_read_paths, forward_read_paths[i])
                    lo_reverse_read_paths <- c(lo_reverse_read_paths, reverse_read_paths[i])
                    lo_filtered_forward_read_paths <- c(lo_filtered_forward_read_paths, filtered_forward_read_paths[i])
                    lo_filtered_reverse_read_paths <- c(lo_filtered_reverse_read_paths, filtered_reverse_read_paths[i])
                }
            }

            leftover_filtering_results = tryFilterAndTrim(
                lo_forward_read_paths,
                lo_filtered_forward_read_paths,
                lo_reverse_read_paths,
                lo_filtered_reverse_read_paths,
                trimLeft = trimLeft,
                trimRight = trimRight,
                truncLen = truncLen,
                maxN = maxN,
                maxEE = maxEE,
                truncQ = truncQ,
                rm_phix = rm_phix,
                compress = compress,
                multithread = multithread,
                matchIDs = TRUE
            )
        },
        finally = {
            return (getFilteringResults(forward_read_paths, filtered_forward_read_paths))
        }
    )
}

main <- function(taskRun) {
    output_path <- builtins$str(ctx_folder_manager$temp)
    taskRun$parameters[["dataset"]]$download()

    forward_read_paths <- c()
    reverse_read_paths <- c()

    for (sample in taskRun$parameters[["dataset"]]$samples) {
        sample$unzip()

        if (startsWith(sample$name, "_metadata") || startsWith(sample$name, "Undetermined")) {
            next
        }

        forward_path <- builtins$str(builtins$list(sample$path$glob(paste0("*", forward_pattern)))[[1]])
        reverse_path <- builtins$str(builtins$list(sample$path$glob(paste0("*", reverse_pattern)))[[1]])

        forward_read_paths <- c(forward_read_paths, forward_path)
        reverse_read_paths <- c(reverse_read_paths, reverse_path)
    }

    # Load metadata file
    metadata <- loadMetadata(taskRun$dataset$getSample("_metadata"))

    # Step 3.2.2: Extract sample names
    sample_names <- c()
    sample_ids <- sapply(strsplit(basename(forward_read_paths), "_"), function(x) x[1])
    for (sample_id in sample_ids) {
        sample_names <- c(sample_names, grep(sample_id, metadata$sampleId, value = TRUE))
    }

    print("Sample names")
    print(sample_names)

    # Step 3.3: Read quality assessment
    print("Step 3.3: Read quality assessment")

    plotQualityProfile(forward_read_paths[1:2])
    plotQualityProfile(reverse_read_paths[1:2])

    quality_profile_plot_path <- file.path(
        output_path,
        "quality_profile_plot.pdf"
    )
    ggsave(quality_profile_plot_path)
    taskRun$createArtifact(
        quality_profile_plot_path,
        basename(quality_profile_plot_path)
    )

    # Step 3.4: Filter and trim:
    print("Step 3.4: Filter and trim")

    #3.4.1 Place filtered read names in a new directory

    filtered_forward_read_paths <- file.path(
        output_path,
        "filtered",
        paste0(sample_names, "_F_filt.fastq.gz")
    )
    names(filtered_forward_read_paths) <- sample_names

    filtered_reverse_read_paths <- file.path(
        output_path,
        "filtered",
        paste0(sample_names, "_R_filt.fastq.gz")
    )
    names(filtered_reverse_read_paths) <- sample_names

    # 3.4.2 Set filtering parameters after checking read quality

    trim_left_forward <- taskRun$parameters[["trimLeftForward"]]
    trim_left_reverse <- taskRun$parameters[["trimLeftReverse"]]
    trim_right_forward <- taskRun$parameters[["trimRightForward"]]
    trim_right_reverse <- taskRun$parameters[["trimRightReverse"]]
    max_n <- taskRun$parameters[["maxN"]]
    max_ee_forward <- taskRun$parameters[["maxEEForward"]]
    max_ee_reverse <- taskRun$parameters[["maxEEReverse"]]
    trunc_q <- taskRun$parameters[["truncQ"]]
    trunc_len_forward <- taskRun$parameters[["truncLenForward"]]
    trunc_len_reverse <- taskRun$parameters[["truncLenReverse"]]

    filtering_results <- tryFilterAndTrim(
        forward_read_paths,
        filtered_forward_read_paths,
        reverse_read_paths,
        filtered_reverse_read_paths,
        trimLeft = c(trim_left_forward, trim_left_reverse),
        trimRight = c(trim_right_forward, trim_right_reverse),
        truncLen = c(trunc_len_forward, trunc_len_reverse),
        maxN = max_n,
        maxEE = c(max_ee_forward, max_ee_reverse),
        truncQ = trunc_q,
        rm_phix = TRUE,
        compress = TRUE,
        multithread = TRUE
    )

    for (path in filtered_forward_read_paths) {
        taskRun$createArtifact(
            path,
            file.path("filtered_reads", basename(path))
        )
    }

    for (path in filtered_reverse_read_paths) {
        taskRun$createArtifact(
            path,
            file.path("filtered_reads", basename(path))
        )
    }

    filtering_results_path <- file.path(output_path, "filtering_results.csv")
    write.csv(
        filtering_results,
        file = filtering_results_path,
        row.names = TRUE
    )
    taskRun$createArtifact(
        filtering_results_path,
        basename(filtering_results_path)
    )

    # Step3.5: DADA2 denoising
    print("Step 3.5: DADA2 denoising")

    # 3.5.1 Learn error rates and plot them
    print("Step 3.5.1: Learn error rates and plot them")

    forward_read_errors <- learnErrors(
        filtered_forward_read_paths,
        multithread = TRUE
    )
    plotErrors(forward_read_errors, nominalQ = TRUE)

    reverse_read_errors <- learnErrors(
        filtered_reverse_read_paths,
        multithread = TRUE
    )
    plotErrors(reverse_read_errors, nominalQ = TRUE)

    errors_plot_path <- file.path(
        output_path,
        "errors_plot.pdf"
    )
    ggsave(errors_plot_path)
    taskRun$createArtifact(
        errors_plot_path,
        basename(errors_plot_path)
    )

    # # 3.5.2 Sample inference
    print("Step 3.5.2: Sample inference")

    dada_forward <- dada(
        filtered_forward_read_paths,
        err = forward_read_errors,
        multithread = TRUE
    )

    dada_reverse <- dada(
        filtered_reverse_read_paths,
        err = reverse_read_errors,
        multithread = TRUE
    )

    # Check the sample inference of the first sample
    print("Forward inference")
    print(dada_forward[[1]])
    print("Reverse inference")
    print(dada_reverse[[1]])

    # 3.5.3 Merge reads
    print("Step 3.5.3: Merge reads")

    merged_reads <- mergePairs(
        dada_forward,
        filtered_forward_read_paths,
        dada_reverse,
        filtered_reverse_read_paths,
        verbose = TRUE
    )
    # Inspect the merger data.frame from the first sample
    print("Merged reads head")
    print(head(merged_reads[[1]]))

    # Step 3.6: ASV table construction
    print("Step 3.6: ASV table construction")

    # FJ: merging the 2 runs together
    seqtab <- makeSequenceTable(merged_reads)
    print("Sequence table dim")
    print(dim(seqtab))

    # Inspect distribution of sequence lengths
    print("Sequence lengths distribution")
    print(table(nchar(getSequences(seqtab))))

    seqtab_path <- file.path(output_path, "ASV_abundance.txt")
    write.table(seqtab, file = seqtab_path)
    taskRun$createArtifact(seqtab_path, "ASV_abundance.txt")

    # Step 3.6.1: Remove chimeras
    print("Step 3.6.1: Remove chimeras")

    # takes around 5 mins
    seqtab_nochim <- removeBimeraDenovo(
        seqtab,
        method = "consensus",
        multithread = TRUE,
        verbose = TRUE
    )

    print("Sequence table non-chimeric dim")
    print(dim(seqtab_nochim))

    print("Sequence table non-chimeric head")
    print(head(seqtab_nochim))

    # Check the frequency of non-chimeras
    print("Frequency of non-chimeras")
    print(sum(seqtab_nochim) / sum(seqtab))

    # Save the sequence file without chimeras as ASV_abundance_nochim_.txt and as an R object
    print("Type of seqtab_nochim")
    print(class(seqtab_nochim))

    # Check if two transpositions are needed.
    seqtab_nochim_path <- file.path(output_path, "ASV_abundance_nochim.Rdata")
    save(seqtab_nochim, file = seqtab_nochim_path)
    taskRun$createArtifact(
        seqtab_nochim_path,
        basename(seqtab_nochim_path)
    )

    seqtab_nochim_path_txt <- file.path(output_path, "ASV_abundance_nochim.txt")
    write.table(
        t(seqtab_nochim),
        file = seqtab_nochim_path_txt,
        col.names = NA,
        row.names = TRUE
    )
    taskRun$createArtifact(
        seqtab_nochim_path_txt,
        basename(seqtab_nochim_path_txt)
    )

    # Step 3.7: Track reads
    print("Step 3.7: Track reads")

    getN <- function(x) {
        return(sum(getUniques(x)))
    }

    track <- cbind(
        filtering_results,
        sapply(dada_forward, getN),
        sapply(dada_reverse, getN),
        sapply(merged_reads, getN),
        rowSums(seqtab_nochim)
    )

    track_matrix <- as.matrix(track)

    colnames(track_matrix) <- c(
        "input", "filtered", "denoisedF", "denoisedR", "merged", "nonchim"
    )
    rownames(track_matrix) <- sample_names

    # Calculate percentages
    track_matrix <- trackPctColumns(
        c("filtered", "denoisedF", "denoisedR", "merged", "nonchim"),
        track_matrix
    )

    track_matrix_path <- file.path(
        output_path,
        paste0(
            "track_trimLeft_trimRight_", trim_left_forward, "_", trim_right_reverse,
            "_maxEE_", max_ee_forward, "_", max_ee_reverse, ".csv"
        )
    )

    write.csv(
        track_matrix,
        file = track_matrix_path,
        row.names = TRUE
    )

    taskRun$createArtifact(
        track_matrix_path,
        basename(track_matrix_path)
    )

    taskRun$createMetrics(list(
        ctx$Metric$create("input", "sample", ctx$MetricType$int, "value", ctx$MetricType$int),
        ctx$Metric$create("nonchim", "sample", ctx$MetricType$int, "value", ctx$MetricType$int),
        ctx$Metric$create("nonchim_pct", "sample", ctx$MetricType$int, "value", ctx$MetricType$percent)
    ))

    for (index in 1:nrow(track_matrix)) {
        input = c(index, track_matrix[index,][["input"]])
        nonchim = c(index, track_matrix[index,][["nonchim"]])
        nonchimPct = c(index, track_matrix[index,][["nonchimPct"]])

        taskRun$submitMetrics(list(
            "input" = input,
            "nonchim" = nonchim,
            "nonchim_pct" = nonchimPct
        ))
    }

    # Step 3.8: Taxonomy assignment using DADA2 or DECIPHER
    print("Step 3.8: Taxonomy assignment using DADA2 or DECIPHER")

    #3.8.1 DECIPHER assign taxonomy

    silva_v138_reference_path <- file.path(
        ctx_folder_manager$cache,
        "SILVA_SSU_r138_2019.RData"
    )

    if (!file.exists(silva_v138_reference_path)) {
        print(paste0("Downloading \"", silva_v138_reference_path, "\"..."))
        # Raise to 10 minutes
        options(timeout = 600)
        download.file(
            url = "http://www2.decipher.codes/Classification/TrainingSets/SILVA_SSU_r138_2019.RData",
            destfile = silva_v138_reference_path
        )
        # Reset to 60 seconds
        options(timeout = 60)
    } else {
        print("Using cached SILVA_SSU_r138_2019.RData")
    }

    # Loaded variable is called "trainingSet"
    load(silva_v138_reference_path)

    # Create a DNAStringSet from the ASVs
    asv_sequences <- getSequences(seqtab_nochim)
    dna_seqtab_nochim <- DNAStringSet(asv_sequences)

    # processors = NULL - use all processors
    ids <- IdTaxa(
        dna_seqtab_nochim,
        trainingSet,
        strand = "top",
        processors = NULL,
        verbose = TRUE
    )

    # ranks of interest
    ranks <- c(
        "domain", "phylum", "class", "order", "family", "genus", "species"
    )

    # Convert the output object of class "Taxa" to a matrix
    # analogous to the output from assignTaxonomy
    taxid_silva <- t(sapply(ids, function(x) {
        m <- match(ranks, x$rank)
        taxa <- x$taxon[m]
        taxa[startsWith(taxa, "unclassified_")]
        taxa
    }))
    colnames(taxid_silva) <- ranks
    rownames(taxid_silva) <- asv_sequences

    taxid_silva_path <- file.path(output_path, "taxid_silva.csv")
    write.csv(
        taxid_silva,
        file = taxid_silva_path,
        col.names = TRUE,
        row.names = TRUE
    )
    taskRun$createArtifact(
        taxid_silva_path,
        basename(taxid_silva_path)
    )

    # Step 3.9: Process files
    print("Step 3.9: Process files")

    prefix <- "asv"
    suffix <- seq(1:length(asv_sequences))
    asv_short_names <- paste(prefix, suffix, sep = "_")
    names(asv_sequences) <- asv_short_names

    asv_shortname_path <- file.path(output_path, "asv_shortname.txt")
    write.table(
        asv_sequences,
        file = asv_shortname_path,
        col.names = NA,
        row.names = TRUE
    )
    taskRun$createArtifact(
        asv_shortname_path,
        basename(asv_shortname_path)
    )

    asv_shortname <- read.table(
        asv_shortname_path,
        sep = "",
        header = TRUE,
        check.names = TRUE
    )

    #Save all asv sequences as a fasta file
    sequences_dna_stringset <- Biostrings::DNAStringSet(asv_sequences)
    sequences_dna_stringset_path <- file.path(
        output_path,
        "asv_sequences.fasta"
    )
    Biostrings::writeXStringSet(
        sequences_dna_stringset,
        file = sequences_dna_stringset_path
    )
    taskRun$createArtifact(
        sequences_dna_stringset_path,
        basename(sequences_dna_stringset_path)
    )

    # Assign asv shortnames to sequences in seqtab_nochim
    seqtab_ps <- as.matrix(t(seqtab_nochim))
    rownames(seqtab_ps) <- asv_short_names

    seqtab_ps_path <- file.path(output_path, "seqtab_ps.RData")
    save(seqtab_ps, file = seqtab_ps_path)
    taskRun$createArtifact(
        seqtab_ps_path,
        basename(seqtab_ps_path)
    )

    ########### 4: MSA ##########
    #MAFFT

    mafft <- function(asv_sequences_path, mafft_output_path) {
        system(paste(
            "mafft",
            asv_sequences_path,
            ">",
            mafft_output_path
        ))
    }

    mafft_output_path <- file.path(output_path, "msa_MAFFT.fasta")
    mafft(
        sequences_dna_stringset_path,
        mafft_output_path
    )
    taskRun$createArtifact(mafft_output_path, basename(mafft_output_path))

    ########### 5: Phylogenetic tree using FASTTREE ##########

    # Make sure to do this, or pseq creation won't work. Takes long (more than 60min)
    fasttree <- function(align_file_path, fasttree_output_path) {
        system(paste(
            "FastTree",
            "-gtr", "-nt",
            align_file_path,
            ">",
            fasttree_output_path
        ))
    }

    tree_file <- file.path(output_path, "phylo_tree_newick_MAFFT1.tree")
    fasttree(mafft_output_path, tree_file)
    taskRun$createArtifact(tree_file, basename(tree_file))

    phylo_tree <- ape::read.tree(tree_file)
    phylo_tree_path <- file.path(output_path, "fasttree_MAFFT1.RData")
    save(phylo_tree, file = phylo_tree_path)
    taskRun$createArtifact(phylo_tree_path, basename(phylo_tree_path))

    ########## 6. Create the phyloseq object ##########
    # To buid the phyloseq object we need the metadata, the seqtab.nochim file, the assigned taxonomy file and the phylogenetic tree.

    # must be TRUE : sample IDs should be the same
    if (!all(rownames(t(seqtab_ps)) %in% metadata$sampleId)) {
        stop(" Sample names must coincide!")
    }
    # Check if metadata is present for every sample
    for (seqname in rownames(t(seqtab_ps))) {
        if (seqname %in% metadata$sampleId) {
        } else {
            stop(paste("missed metadata for :", seqname, " sample"))
        }
    }

    # check if there is a zero
    print("Row sums for seqtab_ps")
    print(rowSums(t(seqtab_ps)))

    # Already done? - Assign asv shortnames to sequences in seqtab.nochim
    seqtab_ps <- as.matrix(t(seqtab_nochim))
    rownames(seqtab_ps) <- asv_short_names

    # Assign asv shortnames to taxonomy table
    rownames(taxid_silva) <- asv_short_names
    print("taxid_silva summary")
    print(summary(taxid_silva))

    taxid_silva_matrix <- as.matrix(taxid_silva)
    print("taxid_silva_matrix head")
    print(head(taxid_silva_matrix))

    taxid_silva_matrix_path <- file.path(output_path, "Taxtable_rownames.csv")
    write.csv(taxid_silva_matrix, file = taxid_silva_matrix_path)
    taskRun$createArtifact(
        taxid_silva_matrix_path,
        basename(taxid_silva_matrix_path)
    )

    # Check the classes of the final files:
    # a. seqtab_ps and taxtab_final must be matrices
    # b. Phylogenetic tree must be a phylo object
    # c. metadata should be a data frame

    print("seqtab_ps type")
    class(seqtab_ps)

    print("taxid_silva_matrix type")
    class(taxid_silva_matrix)

    print("phylo_tree type")
    class(phylo_tree)

    print("metadata type")
    class(metadata)

    # Check if taxa_names between my.tree, taxtab_final and seqtab_ps match
    print("taxa_names comparison")
    taxa_names(phy_tree(phylo_tree))
    taxa_names(tax_table(taxid_silva_matrix))
    taxa_names(otu_table(seqtab_ps, taxa_are_rows = TRUE))

    # Build the phyloseq object
    print("Building the phyloseq object")
    pseq <- phyloseq(
        otu_table(seqtab_ps, taxa_are_rows = TRUE),
        sample_data(metadata),
        tax_table(taxid_silva_matrix),
        phy_tree(phylo_tree)
    )
    sample_data(pseq) <- metadata

    # Save the phyloseq object
    pseq_path <- file.path(output_path, "phyloseq_object.RData")
    save(pseq, file = pseq_path)
    taskRun$createArtifact(pseq_path, basename(pseq_path))

    print("pseq otu_table head")
    print(head(otu_table(pseq)))

    print("pseq tax_table head")
    print(head(tax_table(pseq)))

    print("pseq sample_data head")
    print(head(sample_data(pseq)))

    # Store the phyloseq object into an output dataset
    output_dataset <- ctx$CustomDataset$createDataset(
        paste0(taskRun$id, " - R Step 1: DADA2 and Phyloseq object"),
        taskRun$projectId
    )

    pseq_archive_path <- file.path(output_path, "pseq_archive.zip")
    pseq_archive = zipfile$ZipFile(pseq_archive_path, "w")
    pseq_archive$write(pseq_path, basename(pseq_path))
    pseq_archive$close()

    pseq_sample <- ctx$CustomSample$createCustomSample("phyloseq-object", output_dataset$id, pseq_archive_path)

    output_dataset$finalize()

    taskRun$submitOutput("outputDataset", output_dataset)
}

ctx$initializeRTask(main, args)
