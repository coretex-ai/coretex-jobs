args <- commandArgs(trailingOnly = TRUE)
library(reticulate)

# Import python modules
builtins <- import_builtins()
zipfile <- import("zipfile")
ctx <- import("coretex")
ctx_folder_manager <- import("coretex.folder_manager")

# Contents
# 2. Load packages
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
# Step3.8: Taxonomy assignment using DADA2 assignment (SILVA Db)
# 8.1 Save the Seqtab.RData as a fasta file with ASVs
# 8.2 Save taxonomy table
# Step9: MSA with DECIPHER
# Step10: Phylogenetic tree with Fasttree
# Step11: Create the phyloseq object

loadObjectFromArtifact <- function(artifactsId, path) {
    artifacts <- ctx$Artifact$fetchAll(artifactsId)

    predicate <- function (artifact) { return(artifact$remoteFilePath == path) }
    index <- which(sapply(artifacts, predicate))

    artifact <- artifacts[[index]]

    artifactDir = dirname(builtins$str(artifact$localFilePath))
    if (!file.exists(artifactDir)) {
        dir.create(artifactDir, recursive = TRUE)
    }

    artifact$download()

    return(readRDS(builtins$str(artifact$localFilePath)))
}

main <- function(taskRun) {
    output_path <- builtins$str(ctx_folder_manager$temp)

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
            url = "https://raven.biomech.us:25691/index.php/s/Xriywm4TfT2qmRc/download/SILVA_SSU_r138_2019.binder.RData",
            destfile = silva_v138_reference_path,
            method = "auto"
        )
        # Reset to 60 seconds
        options(timeout = 60)
    } else {
        print("Using cached SILVA_SSU_r138_2019.RData")
    }

    # Loaded variable is called "trainingSet"
    load(silva_v138_reference_path)

    # Create a DNAStringSet from the ASVs
    seqtab_nochim <- loadObjectFromArtifact(taskRun$parameters[["dada2Artifacts"]], "ASV_abundance_nochim.rds")
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
    print("Step 4: MSA")
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
    print("Step 5: Phylogenetic tree using FASTTREE")

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
    print("Step 6: Create the phyloseq object")

    # To buid the phyloseq object we need the metadata, the seqtab.nochim file, the assigned taxonomy file and the phylogenetic tree.
    metadata <- loadObjectFromArtifact(taskRun$parameters[["dada2Artifacts"]], "metadata.rds")

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

    pseq_sample <- output_dataset$add(pseq_archive_path)
    output_dataset$finalize()

    taskRun$submitOutput("outputDataset", output_dataset)
}

ctx$initializeRTask(main, args)
