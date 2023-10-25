args <- commandArgs(trailingOnly = TRUE)
library(reticulate)
py_config()

# Import python modules
builtins <- import_builtins()
ctx <- import("coretex")
ctx_folder_manager <- import("coretex.folder_manager")

#Contents
# 1. Set working directory
# 2. Load packages and phyloseq object
# 3. Pre-process phyloseq object
# 4. Plot read depths
# 5. Rarefaction analyses
# 6. Plot Richness (alpha diversity)
# 7. Normalise the data
# 8. Taxonomic composition

########## 2. Load packages ##########
print("Step 2: Loading packages")
library(remotes)

# I have no idea how to install this using conda
if (!requireNamespace("phyloseq.extended", quietly = TRUE)) {
    print("Installing \"phyloseq.extended\" package")
    remotes::install_github(
        "mahendra-mariadassou/phyloseq-extended",
        ref = "dev",
        quiet = TRUE
    )
}

library(phyloseq)
library(phyloseq.extended)
library(vegan)
library(ggplot2)
library(data.table)
library(forcats)
library(dplyr)
library(RColorBrewer)
library(methods)

loadRData <- function(fileName) {
    load(fileName)
    get(ls()[ls() != "fileName"])
}

loadData <- function(dataset) {
    if (length(dataset$samples) != 1) {
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

perpareSampleData <- function(phyloseqObject, targetColumn) {
    metadata <- data.frame(sample_data(phyloseqObject))
    columnNames <- colnames(metadata)

    print("Checking sample ID column name")
    sampleIdColumn <- getSampleIdColumnName(metadata)
    print(paste("Matched metadata sample ID/name column to", sampleIdColumn))

    print("Renaming metadata sample ID/name column to \"sampleId\"")
    names(metadata)[names(metadata) == sampleIdColumn] <- "sampleId"

    print("Checking body site column name")
    if (!targetColumn %in% columnNames) {
        stop(paste(
            "Entered column name is not present in the phyloseq object. Column names:",
            paste(c(columnNames), collapse = ", ")
        ))
    }

    colnames(metadata)[which(columnNames == targetColumn)] <- "target"
    sample_data(phyloseqObject) <- sample_data(metadata)

    print("Sample Data / Metadata")
    print(colnames(metadata))
    print(head(metadata))

    return(phyloseqObject)
}

subset_samples_custom <- function(pseq, targetColumnValue) {
    metadata <- data.frame(sample_data(pseq))
    if (!targetColumnValue %in% unique(metadata$target)) {
        print(paste("Could not find", targetColumnValue, "among the samples"))
        return(NULL)
    }

    pseq_subset <- pseq
    sample_data(pseq_subset) <- subset(metadata, metadata$target == targetColumnValue)

    return(pseq_subset)
}

all_target_column_values_listed <- function(sampleDataFrame, targetColumnValues) {
    for (uniqueTargetColumnValues in unique(sampleDataFrame$target)) {
        if (!uniqueTargetColumnValues %in% targetColumnValues){
            return (FALSE)
        }
    }

    return (TRUE)
}

genusAbundancePlot <- function(pseq_bac, target_column_value, output_path, taskRun) {
    print(sprintf("Creating abundance plot for %s", target_column_value))

    genus_col_index <- which(rank_names(pseq_bac) == "genus")
    sa <- subset_samples_custom(pseq_bac, target_column_value)

    genus_sum = tapply(taxa_sums(sa), tax_table(sa)[, "genus"], sum, na.rm = FALSE)
    topgenera = names(sort(genus_sum, TRUE))[1:30]

    pseq_top_genera = prune_taxa((tax_table(sa)[, "genus"] %in% topgenera), sa)
    pseq_genera_glom <- tax_glom(pseq_top_genera, taxrank = rank_names(pseq_top_genera)[genus_col_index])
    pseq_genera_melt <- psmelt(pseq_genera_glom)
    ordered_pseq_genera_melt <- setorder(pseq_genera_melt, sampleId, -Abundance)

    file_path <- file.path(output_path, sprintf("pseq_genera_all_%s.csv", target_column_value))
    write.csv(ordered_pseq_genera_melt, file_path)
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    colourCount = 30
    getPalette = colorRampPalette(brewer.pal(12, "Paired"))
    ggplot_pseq_genera_col1 <- ggplot(
            ordered_pseq_genera_melt,
            aes(x = reorder(sampleId, target), y = Abundance, fill = genus)
        ) +
        geom_bar(stat = "identity") +
        facet_wrap(c("target", "sampleId"), ncol = 6, scales = "free") +
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

    abundance_plot_path <- file.path(output_path, sprintf("Abundance_plot_%s.pdf", target_column_value))
    ggsave(
        filename = abundance_plot_path,
        plot = ggplot_pseq_genera_col1,
        width = 300,
        height = 500,
        units = "mm"
    )
    taskRun$createArtifact(abundance_plot_path, paste0("alpha_diversity/", basename(abundance_plot_path)))
    print(sprintf("Uploaded %s", basename(abundance_plot_path)))
}

alphaDiversity <- function(taskRun, pseq, pseq_bac, pseq_bac_normal, output_path) {
    ########### 4. Plot read depths ##########
    print("4. Plot read depths")

    # Build a dataframe containing sample names, read depth, and the variables from the metadata file
    pseq_df <- data.frame(
        read_depth = sample_sums(pseq_bac),
        sample_data(pseq_bac)
    )

    file_path <- file.path(output_path, "pseq_sums.csv")
    write.csv(pseq_df, file = file_path, row.names = FALSE)
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    pseq_samples <- subset_samples(pseq_bac, target != "not_available")
    pseq_df_samples <- data.frame(
        read_depth = sample_sums(pseq_samples),
        sample_data(pseq_samples)
    )

    # Produce a plot to visualise the otu read counts per sample
    read_depth_sum <- ggplot(
        pseq_df, aes(x = sampleId, y = read_depth, fill = target)) +
        geom_bar(stat = "identity") +
        ggtitle("Total Number of Reads") +
        facet_wrap(c("target"), ncol = 8, scales = "free_x") +
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
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    ########### 5. Rarefaction ##########
    print("5. Rarefaction")

    rarefied_curves <- ggrare(pseq_bac, step = 1000, color = "target", se = TRUE)
    rarefied_plot <- rarefied_curves + geom_hline(yintercept = min(sample_sums(pseq_bac)))

    file_path <- file.path(output_path, "Rareraction_plot.pdf")
    ggsave(filename = file_path, plot = rarefied_plot)
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    ########### 6. Plot richness (alpha diversity) ##########
    otu_tab <- otu_table(pseq_bac)

    file_path <- file.path(output_path, "otu_tab.csv")
    write.csv(otu_tab, file_path)
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    Observed_richness <- plot_richness(
        pseq_bac, x = "target",
        color = "target",
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
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
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
    ordered_pseq_new20_melt <- setorder(pseq_new20_melt, sampleId, target)

    ggplot_pseq_new20 <- ggplot(ordered_pseq_new20_melt, aes(x = reorder(sampleId, target), y = Abundance, fill = family))+
        geom_bar(stat = "identity") +
        facet_wrap(. ~target, ncol = 6, scales = "free") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 90, size = 5))
    ggplot_pseq_new20

    file_path <- file.path(output_path, "top20_families_all_target_column_values.pdf")
    ggsave(
        filename = file_path,
        width = 500,
        height = 300,
        units = "mm"
    )
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
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
    ordered_pseq_phyla_melt <- setorder(pseq_phyla_melt, sampleId, target)

    ggplot_pseq_phyla <- ggplot(ordered_pseq_phyla_melt, aes(x = reorder(sampleId, target), y = Abundance, fill = phylum)) +
        geom_bar(stat = "identity") +
        facet_wrap(c("target", "sampleId"), ncol = 9, scales = "free") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 90))

    file_path <- file.path(output_path, "pseq_phyla_plot.pdf")
    ggsave(
        filename = file_path,
        width = 500,
        height = 300,
        units = "mm"
    )
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    colourCount = 20
    getPalette = colorRampPalette(brewer.pal(12, "Paired"))

    file_path <- file.path(output_path, "pseq20families.csv")
    pseq20families <- write.csv(ordered_pseq_new20_melt, file = file_path)
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    ggplot_pseq_new20_col1 <- ggplot(
            ordered_pseq_new20_melt,
            aes(x = reorder(sampleId, target),
            y = Abundance,
            fill = family),
            fill = fct_reorder(family, Abundance)
        ) +
        geom_bar(stat = "identity") +
        facet_wrap(. ~target, ncol = 5, scales = "free") +
        scale_fill_manual(values = colorRampPalette(brewer.pal(12, "Paired"))(colourCount)) +
        theme_bw()
    ggplot_pseq_new20_col1
    p <- ggplot_pseq_new20_col1
    p + theme(
        axis.text.x = element_text(angle = 90, size = 10),
        panel.border = element_rect(colour = "black", fill = NA, size = 1),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
    ) + xlab("target") + ylab("Relative Abundance")

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
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))

    ggplot_pseq_genera <- ggplot(ordered_pseq_genera_melt, aes(x = reorder(Sample, target), y = Abundance, fill = genus))+
        geom_bar(stat = "identity") +
        facet_wrap(c("target", "Sample"), ncol = 4, scales = "free") +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 90, size = 10))
    ggplot_pseq_genera

    colourCount = 30
    getPalette = colorRampPalette(brewer.pal(12, "Paired"))

    ggplot_pseq_genera_col1 <- ggplot(
            ordered_pseq_genera_melt,
            aes(x = reorder(Sample, target),
            y = Abundance, fill = genus)
        ) +
        geom_bar(stat = "identity") +
        facet_wrap(c("target", "Sample"), ncol = 5, scales = "free") +
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
    taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
    print(sprintf("Uploaded %s", basename(file_path)))

    ############### TF: Abundance Plots per sample and bodysite  ##################

    target_column_values <- lapply(taskRun$parameters[["targetColumnValues"]], trimws)

    # Control
    seq_controls <- pseq_bac
    sample_data_frame <- data.frame(sample_data(pseq_bac))

    if (!all_target_column_values_listed(sample_data_frame, target_column_values)){
        # This part is skipped in case all body sites from the metadata file have
        # been entered in the targetColumnValues parameter, because this part performes
        # analysis on all the other body sites that were not entered

        seq_controls_samples <- subset(sample_data_frame, !sample_data_frame$target %in% target_column_values)
        sample_data(seq_controls) <- sample_data(seq_controls_samples)

        file_path <- file.path(output_path, "pseq_control.RData")
        save(seq_controls, file = file_path)
        taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
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
                aes(x = reorder(sampleId, target), y = Abundance, fill = genus)
            ) +
            geom_bar(stat = "identity") +
            facet_wrap(c("target", "sampleId"), ncol = 5, scales = "free") +
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
        taskRun$createArtifact(file_path, paste0("alpha_diversity/", basename(file_path)))
        print(sprintf("Uploaded %s", basename(file_path)))
    }

    # Body sites
    for (target_column_value in target_column_values) {
        genusAbundancePlot(pseq_bac, target_column_value, output_path, taskRun)
    }
}

meltPseqObject <- function(taskRun, pseq, suffix, output_path) {
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
    taskRun$createArtifact(file_path, paste0("beta_diversity/", basename(file_path)))

    print(head(pseq_genera_melt_subset))
    print(count(pseq_genera_melt_subset))
    distinct_sample <- data.frame(lapply(pseq_genera_melt_subset, n_distinct))
    print(class(distinct_sample))

    file_name <- paste0("distinct_", suffix, ".csv")
    file_path <- file.path(output_path, file_name)
    write.csv(distinct_sample, file_path)
    taskRun$createArtifact(file_path, paste0("beta_diversity/", basename(file_path)))
}

betaDiversity <- function(taskRun, pseq, pseq_bac, pseq_bac_normal, output_path) {
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
    taskRun$createArtifact(ordUF_values_path, paste0("beta_diversity/", basename(ordUF_values_path)))

    ordBr_values_path <- file.path(output_path, "ordBr_values_ctrl.csv")
    ordBr_values <- write.csv(ordBr$values, file = ordBr_values_path)
    taskRun$createArtifact(ordBr_values_path, paste0("beta_diversity/", basename(ordBr_values_path)))

    ordBr_vectors_path <- file.path(output_path, "ordBr_vectors_ctrl.csv")
    ordBr_vectors <- write.csv(ordBr$vectors, file = ordBr_vectors_path)
    taskRun$createArtifact(ordBr_vectors_path, paste0("beta_diversity/", basename(ordBr_vectors_path)))

    ordwUF_values_path <- file.path(output_path, "ordwUF_values_ctrl.csv")
    ordwUF_values <- write.csv(ordwUF$values, file = ordwUF_values_path)
    taskRun$createArtifact(ordwUF_values_path, paste0("beta_diversity/", basename(ordwUF_values_path)))

    ordwUF_vectors_path <- file.path(output_path, "ordwUF_vectors_ctrl.csv")
    ordwUF_vectors <- write.csv(ordwUF$vectors, file = ordwUF_vectors_path)
    taskRun$createArtifact(ordwUF_vectors_path, paste0("beta_diversity/", basename(ordwUF_vectors_path)))

    #Check the axes
    plot_scree(ordUF, "Scree Plot: unweighted UniFrac")
    plot_scree(ordwUF, "Scree Plot: weighted UniFrac")
    plot_scree(ordBr, "Scree Plot: Bray Curtis ctrl")
    plot_scree(ordwUF, "Scree Plot: Weighted Unifrac ctrl")

    taget_column_values = sample_data(pseq_bac_normal)[["target"]]

    #Plot for unweighted Unifrac
    PoC_Uni <- plot_ordination(pseq_bac_normal, ordUF, color = "target", shape = "Extraction_protocol") + ggtitle("Unweighted UniFrac") + geom_text(aes(label = taget_column_values), nudge_y = -0.01, size = 3) + geom_point(size = 2)
    PoC_Uni_path = file.path(output_path, "unweighted_Unifrac_plot.pdf")
    ggsave(filename = PoC_Uni_path, plot = PoC_Uni, width = 297, height = 210, units = "mm")
    taskRun$createArtifact(PoC_Uni_path, paste0("beta_diversity/", basename(PoC_Uni_path)))

    #Plot for Bray Curtis
    PoC_Br_PCA <- plot_ordination(pseq_bac_normal, ordBr, color = "target", shape = "Extraction_protocol", axes = c(1,2)) + geom_point(size = 2) + geom_text(aes(label = taget_column_values), nudge_y = -0.01, size = 3)
    PoC_Br_PCA_path <- file.path(output_path, "Bray_curtis_plot.pdf")
    ggsave(filename = PoC_Br_PCA_path, plot = PoC_Br_PCA, width = 297, height = 210, units ="mm")
    taskRun$createArtifact(PoC_Br_PCA_path, paste0("beta_diversity/", basename(PoC_Br_PCA_path)))

    PoC_Br_PCA_1 <- PoC_Br_PCA + theme_bw() +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
    labs(title = "PCoA Bray Curtis", x = "PCo2 (19.3%)", y = "PCo3 (11.2%)")

    PoC_Br_PCA_1_path <- file.path(output_path, "PoC_Br_PCA1.pdf")
    ggsave(filename = PoC_Br_PCA_1_path, plot = PoC_Br_PCA_1, width = 297, height = 210, units = "mm")
    taskRun$createArtifact(PoC_Br_PCA_1_path, paste0("beta_diversity/", basename(PoC_Br_PCA_1_path)))

    #Plot for weighted Unifrac
    PoC_wUF <- plot_ordination(pseq_bac_normal, ordwUF, color = "target", shape = "sex", label = "sample_ID", axes = c(1,2)) + geom_point(size = 2)
    PoC_wUF_1 <- PoC_wUF + theme_bw() +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
    labs(title="PCoA weighted Unifrac" ,x = "PCo3 (8.3%)", y = "PCo4 (4.7%)")
    PoC_wUF_1_path = file.path(output_path, "Plot_PCoA_wUnifrac.pdf")
    ggsave(
        filename = PoC_wUF_1_path,
        plot = PoC_wUF_1,
        width = 297,
        height = 210,
        units = "mm"
    )
    taskRun$createArtifact(PoC_wUF_1_path, paste0("beta_diversity/", basename(PoC_wUF_1_path)))

    # Significance test: test significance with permanova analyses
    #First we generate a dataframe with the metadata
    sampledf <- data.frame(sample_data(pseq_bac_normal))
    print(sampledf)

    # The following code is for datasets with two or more unique body sites
    if (length(unique(sampledf$target)) >= 2) {
        #Next we calculate significance for unweighted Unifrac
        pseq_bac1_p_UF <- adonis2(DistBr ~target, data = sampledf)
        print(pseq_bac1_p_UF)

        #We can also test for differences in dispersion
        #for unweighted unifrac
        betaUF <- betadisper(DistBr, sampledf$target)
        print(permutest(betaUF))
    }

    ##### No of genera, family etc. #####
    # All data
    meltPseqObject(taskRun, pseq_bac_normal, "all", output_path)

    # Use all remaining samples as control
    taget_column_values <- lapply(taskRun$parameters[["targetColumnValues"]], trimws)

    pseq_control <- pseq_bac_normal
    metadata <- data.frame(sample_data(pseq_bac_normal))

    if (!all_target_column_values_listed(metadata, taget_column_values)){
        # This part is skipped in case all body sites from the metadata file have
        # been entered in the targetColumnValues parameter, because this part performes
        # analysis on all the other body sites that were not entered

        sample_data(pseq_control) <- subset(metadata, !metadata$target %in% taget_column_values)
        meltPseqObject(taskRun, pseq_control, "control", output_path)
    }

    for (target_column_value in taget_column_values) {
        target_column_value_pseq = subset_samples_custom(pseq_bac_normal, target_column_value)
        meltPseqObject(taskRun, target_column_value_pseq, target_column_value, output_path)
    }
}

main <- function(taskRun) {
    output_path <- builtins$str(ctx_folder_manager$temp)
    taskRun$dataset$download()

    # Load the phyloseq object
    pseq <- loadData(taskRun$dataset)

    targetColumn <- trimws(taskRun$parameters[["targetColumn"]])
    pseq <- perpareSampleData(pseq, targetColumn)

    pseq_bac <- subset_taxa(pseq, domain == "Bacteria")

    ########## 3. Process phyloseq object ##########
    print("3. Process phyloseq object")

    #Subset data removing non-bacteria
    pseq_bac <- subset_taxa(pseq, domain == "Bacteria")

    #Subset data removing undetermined samples
    pseq_bac <- subset_samples(pseq, !grepl("\\Undetermined", sampleId, fixed = TRUE))

    file_path <- file.path(output_path, "pseq.RData")
    save(pseq_bac, file = file_path)
    taskRun$createArtifact(file_path, basename(file_path))
    print(sprintf("Uploaded %s", basename(file_path)))

    # Normalise the read counts
    pseq_bac_normal = transform_sample_counts(pseq_bac, function(x) x / sum(x))

    file_path <- file.path(output_path, "pseq_normal.RData")
    save(pseq_bac_normal, file = file_path)
    taskRun$createArtifact(file_path, basename(file_path))
    print(sprintf("Uploaded %s", basename(file_path)))

    print("Running: Alpha diversity")
    alphaDiversity(taskRun, pseq, pseq_bac, pseq_bac_normal, ctx_folder_manager$createTempFolder("alpha_diversity"))

    print("Running: Beta diversity")
    betaDiversity(taskRun, pseq, pseq_bac, pseq_bac_normal, ctx_folder_manager$createTempFolder("beta_diversity"))
}

ctx$initializeRTask(main, args)
