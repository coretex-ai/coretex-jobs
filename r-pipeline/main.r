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

# I have no idea how to install this using conda
if (!requireNamespace("phyloseq.extended", quietly = TRUE)) {
    print("Installing \"phyloseq.extended\" package")
    remotes::install_github(
        "mahendra-mariadassou/phyloseq-extended",
        ref = "dev",
        quiet = TRUE
    )
}

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
library(forcats)
library(dplyr)
library(RColorBrewer)
library(methods)

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

dada2_phyloseq <- function(experiment, output_path) {
    forward_read_paths <- c()
    reverse_read_paths <- c()

    for (sample in experiment$dataset$samples) {
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
    metadata <- loadMetadata(experiment$dataset$getSample("_metadata"))

    # Step 3.2.2: Extract sample names
    sample_names <- metadata$sampleId

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
    experiment$createArtifact(
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

    trim_left_forward <- experiment$parameters[["trimLeftForward"]]
    trim_left_reverse <- experiment$parameters[["trimLeftReverse"]]
    trim_right_forward <- experiment$parameters[["trimRightForward"]]
    trim_right_reverse <- experiment$parameters[["trimRightReverse"]]
    max_n <- experiment$parameters[["maxN"]]
    max_ee_forward <- experiment$parameters[["maxEEForward"]]
    max_ee_reverse <- experiment$parameters[["maxEEReverse"]]
    trunc_q <- experiment$parameters[["truncQ"]]
    trunc_len_forward <- experiment$parameters[["truncLenForward"]]
    trunc_len_reverse <- experiment$parameters[["truncLenReverse"]]

    filtering_results <- filterAndTrim(
        forward_read_paths,
        filtered_forward_read_paths,
        reverse_read_paths,
        filtered_reverse_read_paths,
        trimLeft = c(trim_left_forward, trim_left_reverse),
        trimRight = c(trim_right_forward, trim_right_reverse),
        maxN = max_n,
        maxEE = c(max_ee_forward, max_ee_reverse),
        truncQ = trunc_q,
        rm.phix = TRUE,
        compress = TRUE,
        multithread = TRUE
    )

    for (path in filtered_forward_read_paths) {
        experiment$createArtifact(
            path,
            file.path("filtered_reads", basename(path))
        )
    }

    for (path in filtered_reverse_read_paths) {
        experiment$createArtifact(
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
    experiment$createArtifact(
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
    experiment$createArtifact(
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
    experiment$createArtifact(seqtab_path, "ASV_abundance.txt")

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
    experiment$createArtifact(
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
    experiment$createArtifact(
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
            "track_trimLeft_trimRight_", trim_left_forward, "_", trim_right_forward,
            "_maxEE_", max_ee_forward, "_", max_ee_reverse, ".csv"
        )
    )

    write.csv(
        track_matrix,
        file = track_matrix_path,
        row.names = TRUE
    )

    experiment$createArtifact(
        track_matrix_path,
        basename(track_matrix_path)
    )

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
    experiment$createArtifact(
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
    experiment$createArtifact(
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
    experiment$createArtifact(
        sequences_dna_stringset_path,
        basename(sequences_dna_stringset_path)
    )

    # Assign asv shortnames to sequences in seqtab_nochim
    seqtab_ps <- as.matrix(t(seqtab_nochim))
    rownames(seqtab_ps) <- asv_short_names

    seqtab_ps_path <- file.path(output_path, "seqtab_ps.RData")
    save(seqtab_ps, file = seqtab_ps_path)
    experiment$createArtifact(
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
    experiment$createArtifact(mafft_output_path, basename(mafft_output_path))

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
    experiment$createArtifact(tree_file, basename(tree_file))

    phylo_tree <- ape::read.tree(tree_file)
    phylo_tree_path <- file.path(output_path, "fasttree_MAFFT1.RData")
    save(phylo_tree, file = phylo_tree_path)
    experiment$createArtifact(phylo_tree_path, basename(phylo_tree_path))

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
    experiment$createArtifact(
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
    experiment$createArtifact(pseq_path, basename(pseq_path))

    print("pseq otu_table head")
    print(head(otu_table(pseq)))

    print("pseq tax_table head")
    print(head(tax_table(pseq)))

    print("pseq sample_data head")
    print(head(sample_data(pseq)))

    # Store the phyloseq object into an output dataset
    output_dataset <- ctx$CustomDataset$createDataset(
        paste0(experiment$id, " - R Step 1: DADA2 and Phyloseq object"),
        experiment$spaceId
    )

    pseq_archive_path <- file.path(output_path, "pseq_archive.zip")
    pseq_archive = zipfile$ZipFile(pseq_archive_path, "w")
    pseq_archive$write(pseq_path, basename(pseq_path))
    pseq_archive$close()

    pseq_sample <- ctx$CustomSample$createCustomSample("phyloseq-object", output_dataset$id, pseq_archive_path)

    output_dataset$refresh()
    return(output_dataset)
}

loadRData <- function(fileName) {
    load(fileName)
    get(ls()[ls() != "fileName"])
}

loadData <- function(dataset) {
    print(dataset$name)
    print(dataset$id)
    if (dataset$count != 1) {
        stop(">> [Alpha Diversity] Dataset should contain exactly one sample")
    }

    sample <- dataset$samples[[1]]
    sample$unzip()

    rdataInSample <- list.files(path = builtins$str(sample$path), pattern = "\\.RData$", full.names = TRUE)
    if (length(rdataInSample) != 1) {
        stop(">> [Alpha Diversity] Sample should contain exactly one .RData file, which is the phylosec object")
    }

    return(loadRData(rdataInSample[1]))
}

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

perpareSampleData <- function(phyloseqObject, bodySiteColumnName) {
    metadata <- data.frame(sample_data(phyloseqObject))
    columnNames <- colnames(metadata)

    print("Checking sample ID column name")
    sampleIdColumn <- getSampleIdColumnName(metadata)
    print(paste("Matched metadata sample ID/name column to", sampleIdColumn))

    print("Renaming metadata sample ID/name column to \"sampleId\"")
    names(metadata)[names(metadata) == sampleIdColumn] <- "sampleId"

    print("Checking body site column name")
    if (!bodySiteColumnName %in% columnNames) {
        stop(paste(
            "Entered column name is not present in the phyloseq object. Column names:",
            paste(c(columnNames), collapse = ", ")
        ))
    }

    colnames(metadata)[which(columnNames == bodySiteColumnName)] <- "Body_site"
    sample_data(phyloseqObject) <- sample_data(metadata)

    print("Sample Data / Metadata")
    print(colnames(metadata))
    print(head(metadata))

    return(phyloseqObject)
}

subset_samples_custom <- function(pseq, body_site) {
    metadata <- data.frame(sample_data(pseq))
    if (!body_site %in% unique(metadata$Body_site)) {
        print(paste("Could not find", body_site, "among the samples"))
        return(NULL)
    }

    pseq_subset <- pseq
    sample_data(pseq_subset) <- subset(metadata, metadata$Body_site == body_site)

    return(pseq_subset)
}

genusAbundancePlot <- function(pseq_bac, body_site, output_path, experiment) {
    print(sprintf("Creatinng abundance plot for %s", body_site))

    genus_col_index <- which(rank_names(pseq_bac) == "genus")
    sa <- subset_samples_custom(pseq_bac, body_site)

    genus_sum = tapply(taxa_sums(sa), tax_table(sa)[, "genus"], sum, na.rm = FALSE)
    topgenera = names(sort(genus_sum, TRUE))[1:30]

    pseq_top_genera = prune_taxa((tax_table(sa)[, "genus"] %in% topgenera), sa)
    pseq_genera_glom <- tax_glom(pseq_top_genera, taxrank = rank_names(pseq_top_genera)[genus_col_index])
    pseq_genera_melt <- psmelt(pseq_genera_glom)
    ordered_pseq_genera_melt <- setorder(pseq_genera_melt, sampleId, -Abundance)

    file_path <- file.path(output_path, sprintf("pseq_genera_all_%s.csv", body_site))
    write.csv(ordered_pseq_genera_melt, file_path)
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    colourCount = 30
    getPalette = colorRampPalette(brewer.pal(12, "Paired"))
    ggplot_pseq_genera_col1 <- ggplot(
            ordered_pseq_genera_melt,
            aes(x = reorder(sampleId, Body_site), y = Abundance, fill = genus)
        ) +
        geom_bar(stat = "identity") +
        facet_wrap(c("Body_site", "sampleId"), ncol = 6, scales = "free") +
        scale_fill_manual(values = colorRampPalette(brewer.pal(9, "Set1"))(colourCount)) +
        theme_bw() +
        theme(
            axis.text.x = element_text(angle = 90),
            panel.border = element_rect(colour = "black",
            fill = NA, size = 1),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank()
        ) +
        xlab("Figure ID") +
        ylab("Relative Abundance")

    abundance_plot_path <- file.path(output_path, sprintf("Abundance_plot_%s.pdf", body_site))
    ggsave(
        filename = abundance_plot_path,
        plot = ggplot_pseq_genera_col1,
        width = 300,
        height = 500,
        units = "mm"
    )
    experiment$createArtifact(abundance_plot_path, paste0("alpha_diversity/", basename(abundance_plot_path)))
    print(sprintf("Uploaded %s", basename(abundance_plot_path)))
}

alphaDiversity <- function(experiment, pseq, pseq_bac, pseq_bac_normal, output_path) {
    ########### 4. Plot read depths ##########
    print("4. Plot read depths")

    # Build a dataframe containing sample names, read depth, and the variables from the metadata file
    pseq_df <- data.frame(
        read_depth = sample_sums(pseq_bac),
        sample_data(pseq_bac)
    )

    file_path <- file.path(output_path, "pseq_sums.csv")
    write.csv(pseq_df, file = file_path, row.names = FALSE)
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    pseq_samples <- subset_samples(pseq_bac, Body_site != "not_available")
    pseq_df_samples <- data.frame(
        read_depth = sample_sums(pseq_samples),
        sample_data(pseq_samples)
    )

    # Produce a plot to visualise the otu read counts per sample
    read_depth_sum <- ggplot(
        pseq_df, aes(x = sampleId, y = read_depth, fill = Body_site)) +
        geom_bar(stat = "identity") +
        ggtitle("Total Number of Reads") +
        facet_wrap(c("Body_site"), ncol = 8, scales = "free_x") +
        theme(axis.text.x = element_text(angle = 90)) +
        labs(x = "Samples by Body site") +
        labs(y = "Number of Reads")

    file_path <- file.path(output_path, "Plot_Read_Counts_allsamples.pdf")
    ggsave(
        filename = file_path,
        plot = read_depth_sum,
        width = 450,
        height = 300,
        units = "mm"
    )
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    ########### 5. Rarefaction ##########
    print("5. Rarefaction")

    rarefied_curves <- ggrare(pseq_bac, step = 1000, color = "Body_site", se = TRUE)
    rarefied_plot <- rarefied_curves + geom_hline(yintercept = min(sample_sums(pseq_bac)))

    file_path <- file.path(output_path, "Rareraction_plot.pdf")
    ggsave(filename = file_path, plot = rarefied_plot)
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    ########### 6. Plot richness (alpha diversity) ##########
    otu_tab <- otu_table(pseq_bac)

    file_path <- file.path(output_path, "otu_tab.csv")
    write.csv(otu_tab, file_path)
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    Observed_richness <- plot_richness(
        pseq_bac, x = "Body_site",
        color = "Body_site",
        measures = c("Observed", "Shannon")
    )
    Observed_richness_plot <- plot(Observed_richness + geom_boxplot())

    file_path <- file.path(output_path, "Observed_richness_plot.pdf")
    ggsave(
        filename = file_path,
        plot = Observed_richness_plot,
        width = 450,
        height = 300,
        units = "mm"
    )
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    ########## 7. Taxonomic composition ##########
    print("8. Taxonomic composition")

    ######### LW: 7.1 Plot families ##########
    family_sum = tapply(taxa_sums(pseq_bac_normal), tax_table(pseq_bac_normal)[, "family"], sum, na.rm=FALSE)
    top20family = names(sort(family_sum, TRUE))[1:20]

    pseq_new_top20 = prune_taxa((tax_table(pseq_bac_normal)[, "family"] %in% top20family), pseq_bac_normal)

    family_col_index <- which(rank_names(pseq_new_top20) == "family")
    pseq_new20_glom <- tax_glom(pseq_new_top20, taxrank = rank_names(pseq_new_top20)[family_col_index])

    pseq_new20_melt <- psmelt(pseq_new20_glom)

    ordered_pseq_new20_melt <- setorder(pseq_new20_melt, sampleId, -Abundance)
    ordered_pseq_new20_melt <- setorder(pseq_new20_melt, sampleId, Body_site)

    ggplot_pseq_new20 <- ggplot(ordered_pseq_new20_melt, aes(x = reorder(sampleId, Body_site), y = Abundance, fill = family))+
        geom_bar(stat = "identity") +
        facet_wrap(. ~Body_site, ncol = 6, scales = "free") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 90, size = 5))
    ggplot_pseq_new20

    file_path <- file.path(output_path, "top20_families_allbodysites.pdf")
    ggsave(
        filename = file_path,
        width = 500,
        height = 300,
        units = "mm"
    )
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    ########## LW: 7.2 Plot phyla #########
    print("8.2 Plot phyla")

    Phyla_sum = tapply(taxa_sums(pseq_bac_normal), tax_table(pseq_bac_normal)[, "phylum"], sum, na.rm = FALSE)
    top10phyla = names(sort(Phyla_sum, TRUE))[1:10]

    pseq_phyla2 = prune_taxa((tax_table(pseq_bac_normal)[, "phylum"] %in% top10phyla), pseq_bac_normal)

    phylum_col_index <- which(rank_names(pseq_bac_normal) == "phylum")
    pseq_phyla_glom <- tax_glom(pseq_bac_normal, taxrank = rank_names(pseq_bac_normal)[phylum_col_index])

    pseq_phyla_melt <- psmelt(pseq_phyla_glom)

    ordered_pseq_phyla_melt <- setorder(pseq_phyla_melt, sampleId, -Abundance)
    ordered_pseq_phyla_melt <- setorder(pseq_phyla_melt, sampleId, Body_site)

    ggplot_pseq_phyla <- ggplot(ordered_pseq_phyla_melt, aes(x = reorder(sampleId, Body_site), y = Abundance, fill = phylum)) +
        geom_bar(stat = "identity") +
        facet_wrap(c("Body_site", "sampleId"), ncol = 9, scales = "free") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 90))

    file_path <- file.path(output_path, "pseq_phyla_plot.pdf")
    ggsave(
        filename = file_path,
        width = 500,
        height = 300,
        units = "mm"
    )
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    colourCount = 20
    getPalette = colorRampPalette(brewer.pal(12, "Paired"))

    file_path <- file.path(output_path, "pseq20families.csv")
    pseq20families <- write.csv(ordered_pseq_new20_melt, file = file_path)
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    ggplot_pseq_new20_col1 <- ggplot(
            ordered_pseq_new20_melt,
            aes(x = reorder(sampleId, Body_site),
            y = Abundance,
            fill = family),
            fill = fct_reorder(family, Abundance)
        ) +
        geom_bar(stat = "identity") +
        facet_wrap(. ~Body_site, ncol = 5, scales = "free") +
        scale_fill_manual(values = colorRampPalette(brewer.pal(12, "Paired"))(colourCount)) +
        theme_bw()
    ggplot_pseq_new20_col1
    p <- ggplot_pseq_new20_col1
    p + theme(
        axis.text.x = element_text(angle = 90, size = 10),
        panel.border = element_rect(colour = "black", fill = NA, size = 1),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
    ) + xlab("Body_site") + ylab("Relative Abundance")

    ############# LW: 7.3 Plot genera ###########
    print("8.3 Plot genera")

    genus_sum = tapply(taxa_sums(pseq_bac_normal), tax_table(pseq_bac_normal)[, "genus"], sum, na.rm = FALSE)
    topgenera = names(sort(genus_sum, TRUE))[1:30]

    pseq_top_genera <- prune_taxa((tax_table(pseq_bac_normal)[, "genus"] %in% topgenera), pseq_bac_normal)

    genus_col_index <- which(rank_names(pseq_bac_normal) == "genus")
    pseq_genera_glom <- tax_glom(pseq_top_genera, taxrank = rank_names(pseq_top_genera)[genus_col_index])

    pseq_genera_melt <- psmelt(pseq_genera_glom)

    ordered_pseq_genera_melt <- setorder(pseq_genera_melt, sampleId, -Abundance)

    file_path <- file.path(output_path, "pseq_genera_melt.csv")
    write.csv(ordered_pseq_genera_melt, file_path)
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    ggplot_pseq_genera <- ggplot(ordered_pseq_genera_melt, aes(x = reorder(Sample, Body_site), y = Abundance, fill = genus))+
        geom_bar(stat = "identity") +
        facet_wrap(c("Body_site", "Sample"), ncol = 4, scales = "free") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 90, size = 10))
    ggplot_pseq_genera

    colourCount = 30
    getPalette = colorRampPalette(brewer.pal(12, "Paired"))

    ggplot_pseq_genera_col1 <- ggplot(
            ordered_pseq_genera_melt,
            aes(x = reorder(Sample, Body_site),
            y = Abundance, fill = genus)
        ) +
        geom_bar(stat = "identity") +
        facet_wrap(c("Body_site", "Sample"), ncol = 5, scales = "free") +
        scale_fill_manual(values = colorRampPalette(brewer.pal(12, "Paired"))(colourCount)) +
        theme_bw() +
        theme(
            axis.text.x = element_text(angle = 90),
            panel.border = element_rect(colour = "black", fill = NA, size = 1),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank()) + xlab("Sample ID") + ylab("Relative Abundance")
    ggplot_pseq_genera_col1

    file_path <- file.path(output_path, "Plot_Abundancde_30_all.pdf")
    ggsave(
        filename = file_path,
        plot = ggplot_pseq_genera_col1,
        width = 400,
        height = 700,
        units = "mm"
    )
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    ############### TF: Abundance Plots per sample and bodysite  ##################

    body_sites <- unlist(experiment$parameters["bodySites"])

    # Control
    seq_controls <- pseq_bac
    sample_data_frame <- data.frame(sample_data(pseq_bac))
    seq_controls_samples <- subset(sample_data_frame, !sample_data_frame$Body_site %in% body_sites)
    sample_data(seq_controls) <- sample_data(seq_controls_samples)

    file_path <- file.path(output_path, "pseq_control.RData")
    save(seq_controls, file = file_path)
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    genus_sum = tapply(taxa_sums(seq_controls), tax_table(seq_controls)[, "genus"], sum, na.rm=FALSE)
    topgenera = names(sort(genus_sum, TRUE))[1:30]
    pseq_top_genera = prune_taxa((tax_table(seq_controls)[, "genus"] %in% topgenera), seq_controls)
    pseq_genera_glom <- tax_glom(pseq_top_genera, taxrank = rank_names(pseq_top_genera)[genus_col_index])
    pseq_genera_melt <- psmelt(pseq_genera_glom)
    ordered_pseq_genera_melt <- setorder(pseq_genera_melt, sampleId, -Abundance)
    rank_names(seq_controls)

    display.brewer.all()
    colourCount = 30
    getPalette = colorRampPalette(brewer.pal(12, "Paired"))
    ggplot_pseq_genera_col1 <- ggplot(
            ordered_pseq_genera_melt,
            aes(x = reorder(sampleId, Body_site), y = Abundance, fill = genus)
        ) +
        geom_bar(stat = "identity") +
        facet_wrap(c("Body_site", "sampleId"), ncol = 5, scales = "free") +
        scale_fill_manual(values = colorRampPalette(brewer.pal(11, "Paired"))(colourCount)) +
        theme_bw() +
        theme(
            axis.text.x = element_text(angle = 90),
            panel.border = element_rect(colour = "black", fill = NA, size = 1),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank()
        ) +
        xlab("Figure ID") + ylab("Relative Abundance")
    ggplot_pseq_genera_col1

    file_path <- file.path(output_path, "Plot_Abundance_30_Controls.pdf")
    ggsave(
        filename = file_path,
        plot = ggplot_pseq_genera_col1,
        width = 350,
        height = 500,
        units = "mm"
    )
    experiment$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    # Body sites
    for (bodySite in body_sites) {
        genusAbundancePlot(pseq_bac, bodySite, output_path, experiment)
    }
}

meltPseqObject <- function(experiment, pseq, suffix, output_path) {
    print(paste0("Beta Diversity: Analyzing ", suffix))
    genus_col_index <- which(rank_names(pseq) == "genus")

    genus_sum = tapply(taxa_sums(pseq), tax_table(pseq)[, "genus"], sum, na.rm=FALSE)
    topgenera = names(sort(genus_sum, TRUE))
    pseq_top_genera = prune_taxa((tax_table(pseq)[, "genus"] %in% topgenera), pseq)
    pseq_genera_glom <- tax_glom(pseq_top_genera, taxrank = rank_names(pseq_top_genera)[genus_col_index])
    pseq_genera_melt <- psmelt(pseq_genera_glom)
    pseq_genera_melt_subset <- subset(pseq_genera_melt, Abundance != 0, )
    ordered_pseq_genera_melt <- setorder(pseq_genera_melt_subset, sampleId, -Abundance)

    file_name <- paste0("pseq_genera_melt_", suffix, "_allgenera.csv")
    file_path <- file.path(output_path, file_name)
    write.csv(ordered_pseq_genera_melt, file_path)
    experiment$createArtifact(file_path, paste0("beta_diversity/", basename(file_path)))

    print(head(pseq_genera_melt_subset))
    print(count(pseq_genera_melt_subset))
    distinct_sample <- data.frame(lapply(pseq_genera_melt_subset, n_distinct))
    print(class(distinct_sample))

    file_name <- paste0("distinct_", suffix, ".csv")
    file_path <- file.path(output_path, file_name)
    write.csv(distinct_sample, file_path)
    experiment$createArtifact(file_path, paste0("beta_diversity/", basename(file_path)))
}

betaDiversity <- function(experiment, pseq, pseq_bac, pseq_bac_normal, output_path) {
    #Calculate distances
    DistUF = phyloseq::distance(pseq_bac_normal, method = "unifrac")
    DistwUF = phyloseq::distance(pseq_bac_normal, method = "wunifrac")
    DistBr = phyloseq::distance(pseq_bac_normal, method = "bray")

    #Ordinate
    ordUF = ordinate(pseq_bac_normal, method = "PCoA", distance = DistUF)
    ordwUF = ordinate(pseq_bac_normal, method = "PCoA", distance = DistwUF)
    ordBr = ordinate(pseq_bac_normal, method = "PCoA", distance = DistBr)

    ordUF_values_path <- file.path(output_path, "ordUF_values_ctrl.csv")
    ordUF_values <- write.csv(ordUF$values, file = ordUF_values_path)
    experiment$createArtifact(ordUF_values_path, paste0("beta_diversity/", basename(ordUF_values_path)))

    ordBr_values_path <- file.path(output_path, "ordBr_values_ctrl.csv")
    ordBr_values <- write.csv(ordBr$values, file = ordBr_values_path)
    experiment$createArtifact(ordBr_values_path, paste0("beta_diversity/", basename(ordBr_values_path)))

    ordBr_vectors_path <- file.path(output_path, "ordBr_vectors_ctrl.csv")
    ordBr_vectors <- write.csv(ordBr$vectors, file = ordBr_vectors_path)
    experiment$createArtifact(ordBr_vectors_path, paste0("beta_diversity/", basename(ordBr_vectors_path)))

    ordwUF_values_path <- file.path(output_path, "ordwUF_values_ctrl.csv")
    ordwUF_values <- write.csv(ordwUF$values, file = ordwUF_values_path)
    experiment$createArtifact(ordwUF_values_path, paste0("beta_diversity/", basename(ordwUF_values_path)))

    ordwUF_vectors_path <- file.path(output_path, "ordwUF_vectors_ctrl.csv")
    ordwUF_vectors <- write.csv(ordwUF$vectors, file = ordwUF_vectors_path)
    experiment$createArtifact(ordwUF_vectors_path, paste0("beta_diversity/", basename(ordwUF_vectors_path)))

    #Check the axes
    plot_scree(ordUF, "Scree Plot: unweighted UniFrac")
    plot_scree(ordwUF, "Scree Plot: weighted UniFrac")
    plot_scree(ordBr, "Scree Plot: Bray Curtis ctrl")
    plot_scree(ordwUF, "Scree Plot: Weighted Unifrac ctrl")

    body_sites = sample_data(pseq_bac_normal)[["Body_site"]]

    #Plot for unweighted Unifrac
    PoC_Uni <- plot_ordination(pseq_bac_normal, ordUF, color = "Body_site", shape = "Extraction_protocol") + ggtitle("Unweighted UniFrac") + geom_text(aes(label = body_sites), nudge_y = -0.01, size = 3) + geom_point(size=2)
    PoC_Uni_path = file.path(output_path, "unweighted_Unifrac_plot.pdf")
    ggsave(filename = PoC_Uni_path, plot = PoC_Uni, width = 297, height = 210, units = "mm")
    experiment$createArtifact(PoC_Uni_path, paste0("beta_diversity/", basename(PoC_Uni_path)))

    #Plot for Bray Curtis
    PoC_Br_PCA <- plot_ordination(pseq_bac_normal, ordBr, color = "Body_site", shape = "Extraction_protocol", axes = c(1,2)) + geom_point(size = 2) + geom_text(aes(label = body_sites), nudge_y = -0.01, size = 3)
    PoC_Br_PCA_path <- file.path(output_path, "Bray_curtis_plot.pdf")
    ggsave(filename = PoC_Br_PCA_path, plot = PoC_Br_PCA, width = 297, height = 210, units ="mm")
    experiment$createArtifact(PoC_Br_PCA_path, paste0("beta_diversity/", basename(PoC_Br_PCA_path)))

    PoC_Br_PCA_1 <- PoC_Br_PCA + theme_bw() +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
    labs(title="PCoA Bray Curtis",x ="PCo2 (19.3%)", y = "PCo3 (11.2%)")

    PoC_Br_PCA_1_path <- file.path(output_path, "PoC_Br_PCA1.pdf")
    ggsave(filename = PoC_Br_PCA_1_path, plot = PoC_Br_PCA_1, width = 297, height = 210, units = "mm")
    experiment$createArtifact(PoC_Br_PCA_1_path, paste0("beta_diversity/", basename(PoC_Br_PCA_1_path)))

    #Plot for weighted Unifrac
    PoC_wUF <- plot_ordination(pseq_bac_normal, ordwUF, color="Body_site", shape = "sex", label = "sample_ID", axes = c(3,4)) + geom_point(size = 2)
    PoC_wUF_1 <- PoC_wUF + theme_bw() +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
    labs(title="PCoA weighted Unifrac",x ="PCo3 (8.3%)", y = "PCo4 (4.7%)")
    PoC_wUF_1_path = file.path(output_path, "Plot_PCoA_wUnifrac.pdf")
    ggsave(
        filename = PoC_wUF_1_path,
        plot = PoC_wUF_1,
        width = 297,
        height = 210,
        units = "mm"
    )
    experiment$createArtifact(PoC_wUF_1_path, paste0("beta_diversity/", basename(PoC_wUF_1_path)))

    # Significance test: test significance with permanova analyses
    #First we generate a dataframe with the metadata
    sampledf <- data.frame(sample_data(pseq_bac_normal))
    print(sampledf)

    #Next we calculate significance for unweighted Unifrac
    pseq_bac1_p_UF <- adonis2(DistBr ~Body_site, data = sampledf)
    print(pseq_bac1_p_UF)

    #We can also test for differences in dispersion
    #for unweighted unifrac
    betaUF <- betadisper(DistBr, sampledf$Body_site)
    print(permutest(betaUF))

    ##### No of genera, family etc. #####
    # All data
    meltPseqObject(experiment, pseq_bac_normal, "all", output_path)

    # Use all remaining samples as control
    pseq_control <- pseq_bac_normal
    metadata <- data.frame(sample_data(pseq_bac_normal))
    sample_data(pseq_control) <- subset(metadata, !metadata$Body_site %in% experiment$parameters[["bodySites"]])

    meltPseqObject(experiment, pseq_control, "control", output_path)

    for (body_site in experiment$parameters[["bodySites"]]) {
        body_site_pseq = subset_samples_custom(pseq_bac_normal, body_site)
        meltPseqObject(experiment, body_site_pseq, body_site, output_path)
    }
}

main <- function(experiment) {
    output_path <- builtins$str(ctx_folder_manager$temp)
    experiment$dataset$download()

    # Step 1: DADA2 + Phyloseq object
    phyloseq_dataset <- dada2_phyloseq(experiment, output_path)
    phyloseq_dataset$download()

    # Load the phyloseq object
    pseq <- loadData(phyloseq_dataset)

    bodySiteColumnName <- experiment$parameters[["bodySiteColumnName"]]
    pseq <- perpareSampleData(pseq, bodySiteColumnName)

    pseq_bac <- subset_taxa(pseq, domain == "Bacteria")

    ########## 3. Process phyloseq object ##########
    print("3. Process phyloseq object")

    #Subset data removing non-bacteria
    pseq_bac <- subset_taxa(pseq, domain == "Bacteria")

    #Subset data removing undetermined samples
    pseq_bac <- subset_samples(pseq, !grepl("\\Undetermined", sampleId, fixed = TRUE))

    file_path <- file.path(output_path, "pseq.RData")
    save(pseq_bac, file = file_path)
    experiment$createArtifact(file_path, basename(file_path))
    print(sprintf("Uploaded %s", basename(file_path)))

    # Normalise the read counts
    pseq_bac_normal = transform_sample_counts(pseq_bac, function(x) x / sum(x))

    file_path <- file.path(output_path, "pseq_normal.RData")
    save(pseq_bac_normal, file = file_path)
    experiment$createArtifact(file_path, basename(file_path))
    print(sprintf("Uploaded %s", basename(file_path)))

    print("Running: Alpha diversity")
    alphaDiversity(experiment, pseq, pseq_bac, pseq_bac_normal, ctx_folder_manager$createTempFolder("alpha_diversity"))

    print("Running: Beta diversity")
    betaDiversity(experiment, pseq, pseq_bac, pseq_bac_normal, ctx_folder_manager$createTempFolder("beta_diversity"))
}

ctx$initializeRTask(main, args)
