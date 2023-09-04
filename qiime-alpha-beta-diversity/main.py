from pathlib import Path
from zipfile import ZipFile

import os
import csv
import logging

from coretex import CustomDataset, CustomSample, Run, folder_manager
from coretex.job import initializeJob
from coretex.bioinformatics import CommandException, ctx_qiime2


def columnNamePresent(metadataPath: Path, columnName: str) -> bool:
    with metadataPath.open("r") as metadata:
        for row in csv.reader(metadata, delimiter = "\t"):
            return columnName in row

    raise RuntimeError(">> [Qiime: Alpha & Beta Diversity] Metadata file is empty")


def diversityCoreMetricsPhylogeneticSample(
    sample: CustomSample,
    tablePath: Path,
    samplingDepth: int,
    metadataPath: Path,
    outputDir: Path
) -> Path:

    phylogenyPath = sample.joinPath("rooted-tree.qza")
    outputPath = outputDir / "core-metrics-results"

    ctx_qiime2.diversityCoreMetricsPhylogenetic(
        str(phylogenyPath),
        str(tablePath),
        samplingDepth,
        str(metadataPath),
        str(outputPath)
    )

    outputZipPath = outputDir / f"{outputPath.name}.zip"
    with ZipFile(outputZipPath, "w") as outputFile:
        for path in outputPath.glob("**/*"):
            outputFile.write(path, path.name)

    return outputZipPath


def diversityAlphaGroupSignificance(
    alphaDiversityPath: Path,
    metadataPath: Path,
    sampleIndex: int,
    outputDatasetId: int,
    outputPath: Path,
    run: Run
):

    ctx_qiime2.diversityAlphaGroupSignificance(
        str(alphaDiversityPath),
        str(metadataPath),
        str(outputPath)
    )

    ctx_qiime2.createSample(
        f"{sampleIndex}-{outputPath.stem}",
        outputDatasetId,
        outputPath,
        run,
        "Step 7: Alpha & Beta diversity analysis"
    )


def diversityBetaGroupSignificance(
    distanceMatrixPath: Path,
    metadataPath: Path,
    metadataColumn: str,
    sampleIndex: int,
    outputDatasetId: int,
    outputPath: Path,
    run: Run
):

    ctx_qiime2.diversityBetaGroupSignificance(
        str(distanceMatrixPath),
        str(metadataPath),
        metadataColumn,
        str(outputPath),
        True
    )

    ctx_qiime2.createSample(
        f"{sampleIndex}-{outputPath.stem}",
        outputDatasetId,
        outputPath,
        run,
        "Step 7: Alpha & Beta diversity analysis"
    )


def emperorPlot(
    pcoaPath: Path,
    metadataPath: Path,
    sampleIndex: int,
    outputDatasetId: int,
    outputPath: Path,
    run: Run
):

    ctx_qiime2.emperorPlot(
        str(pcoaPath),
        str(metadataPath),
        "days-since-experiment-start",
        str(outputPath)
    )

    ctx_qiime2.createSample(
        f"{sampleIndex}-{outputPath.stem}",
        outputDatasetId,
        outputPath,
        run,
        "Step 7: Alpha & Beta diversity analysis"
    )


def diversityAlphaRarefaction(
    tablePath: Path,
    phylogenyPath: Path,
    maxDepth: int,
    metadataPath: Path,
    sampleIndex: int,
    outputDatasetId: int,
    outputPath: Path,
    run: Run
):
    ctx_qiime2.diversityAlphaRarefaction(
        str(tablePath),
        str(phylogenyPath),
        maxDepth,
        str(metadataPath),
        str(outputPath)
    )

    ctx_qiime2.createSample(
        f"{sampleIndex}-{outputPath.stem}",
        outputDatasetId,
        outputPath,
        run,
        "Step 7: Alpha & Beta diversity analysis"
    )


def processSample(
    index: int,
    sample: CustomSample,
    metadataSample: CustomSample,
    denoisedSample: CustomSample,
    run: Run,
    outputDataset: CustomDataset,
    outputDir: Path
):

    sample.unzip()
    metadataSample.unzip()
    denoisedSample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    metadataPath = ctx_qiime2.getMetadata(metadataSample)
    targetTypeColumn = run.parameters["targetTypeColumn"]

    if not columnNamePresent(metadataPath, targetTypeColumn):
        logging.error(f">> [Qiime: Alpha & Beta Diversity] targetTypeColumn")

    # First step:
    # Apply the core-metrics-phylogenetic method, which rarefies a
    # FeatureTable[Frequency] to a user-specified depth, computes
    # several alpha and beta diversity metrics, and generates principle
    # coordinates analysis (PCoA) plots using Emperor for each
    # of the beta diversity metrics.
    logging.info(">> [Qiime: Alpha & Beta Diversity] Apllying the core-metrics-phylogenetic method")
    try:
        coreMetricsPath = diversityCoreMetricsPhylogeneticSample(
            sample,
            denoisedSample.joinPath("table.qza"),
            run.parameters["samplingDepth"],
            metadataPath,
            sampleOutputDir
        )

        coreMetricsSample = ctx_qiime2.createSample(
            f"{index}-core-metrics-phylogenetic",
            outputDataset.id,
            coreMetricsPath,
            run,
            "Step 7: Alpha & Beta diversity analysis"
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to execute \"qiime diversity core-metrics-phylogenetic\"")

    # Second step:
    # Explore the microbial composition of the samples in the context of the sample metadata
    coreMetricsSample.download()
    coreMetricsSample.unzip()

    logging.info(">> [Qiime: Alpha & Beta Diversity] Generating faith_pd_vector.qza")
    try:
        diversityAlphaGroupSignificance(
            coreMetricsSample.joinPath("faith_pd_vector.qza"),
            metadataPath,
            index,
            outputDataset.id,
            sampleOutputDir / "faith-pd-group-significance.qzv",
            run
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create faith_pd_vector.qza")

    logging.info(">> [Qiime: Alpha & Beta Diversity] Generating evenness_vector.qza")
    try:
        diversityAlphaGroupSignificance(
            coreMetricsSample.joinPath("evenness_vector.qza"),
            metadataPath,
            index,
            outputDataset.id,
            sampleOutputDir / "evenness-group-significance.qzv",
            run
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create evenness_vector.qza")

    # Third step:
    # Analyze sample composition in the context of categorical metadata using PERMANOVA
    # Test whether distances between samples within a group, such as
    # samples from the same body site (e.g., gut), are more similar to
    # each other then they are to samples from the other
    # groups (e.g., tongue, left palm, and right palm).
    logging.info(">> [Qiime: Alpha & Beta Diversity] Generating unweighted_unifrac_distance_matrix.qza")
    try:
        diversityBetaGroupSignificance(
            coreMetricsSample.joinPath("unweighted_unifrac_distance_matrix.qza"),
            metadataPath,
            targetTypeColumn,
            index,
            outputDataset.id,
            sampleOutputDir / "unweighted-unifrac-body-site-significance.qzv",
            run
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create unweighted_unifrac_distance_matrix.qza")

    try:
        diversityBetaGroupSignificance(
            coreMetricsSample.joinPath("unweighted_unifrac_distance_matrix.qza"),
            metadataPath,
            targetTypeColumn,
            index,
            outputDataset.id,
            sampleOutputDir / "unweighted-unifrac-subject-group-significance.qzv",
            run
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create unweighted_unifrac_distance_matrix.qza")

    # Fourth step:
    # Exploring microbial community composition in the context of sample metadata using ordination
    logging.info(">> [Qiime: Alpha & Beta Diversity] Generating unweighted_unifrac_pcoa_results.qza")
    try:
        emperorPlot(
            coreMetricsSample.joinPath("unweighted_unifrac_pcoa_results.qza"),
            metadataPath,
            index,
            outputDataset.id,
            sampleOutputDir / "unweighted-unifrac-emperor-days-since-run-start.qzv",
            run
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create unweighted-unifrac-emperor-days-since-experiment-start.qzv")

    logging.info(">> [Qiime: Alpha & Beta Diversity] Generating bray_curtis_pcoa_results.qza")
    try:
        emperorPlot(
            coreMetricsSample.joinPath("bray_curtis_pcoa_results.qza"),
            metadataPath,
            index,
            outputDataset.id,
            sampleOutputDir / "bray-curtis-emperor-days-since-experiment-start.qzv",
            run
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create bray-curtis-emperor-days-since-experiment-start.qzv")

    # Fifth step:
    # This visualizer computes one or more alpha diversity metrics at multiple sampling depths,
    # in steps between 1 and the value provided as --p-max-depth
    logging.info(">> [Qiime: Alpha & Beta Diversity] Generating table.qza")
    try:
        diversityAlphaRarefaction(
            denoisedSample.joinPath("table.qza"),
            sample.joinPath("rooted-tree.qza"),
            run.parameters["maxDepth"],
            metadataPath,
            index,
            outputDataset.id,
            sampleOutputDir / "alpha-rarefaction.qzv",
            run
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create alpha-rarefaction.qzv")


def main(run: Run[CustomDataset]):
    # If GPU is detected but not configured properly we have
    # to disable its usage for unifrac otherwise run
    # will crash
    try:
        from py3nvml import py3nvml
        py3nvml.nvmlInit()

        os.environ["UNIFRAC_USE_GPU"] = "Y"
        logging.info(">> [Qiime: Alpha & Beta Diversity] GPU will be used for \"unifrac\" calculations")

        py3nvml.nvmlShutdown()
    except:
        os.environ["UNIFRAC_USE_GPU"] = "N"

        logging.warning(">> [Qiime: Alpha & Beta Diversity] GPU will not be used for \"unifrac\" calculations")

    run.dataset.download()

    phylogeneticTreeSamples = ctx_qiime2.getPhylogeneticTreeSamples(run.dataset)
    if len(phylogeneticTreeSamples) == 0:
        raise ValueError(">> [Qiime: Alpha & Beta Diversity] Dataset has 0 phylogenetic tree samples")

    importedDataset: CustomDataset = run.parameters["importedDataset"]
    importedDataset.download()

    denoisedDataset: CustomDataset = run.parameters["denoisedDataset"]
    denoisedDataset.download()

    outputDir = folder_manager.createTempFolder("alpha_beta_output")
    outputDataset = CustomDataset.createDataset(
        f"{run.id} - Step 7: Alpha & Beta diversity",
        run.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Qiime: Alpha & Beta Diversity] Failed to create output dataset")

    for sample in phylogeneticTreeSamples:
        index = ctx_qiime2.sampleNumber(sample)

        metadataSample = importedDataset.getSample(f"{index}-metadata")
        if metadataSample is None:
            raise ValueError(f">> [Qiime: Alpha & Beta Diversity] metadata sample not found")

        denoisedSample = denoisedDataset.getSample(f"{index}-denoise")
        if denoisedSample is None:
            raise ValueError(f">> [Qiime: Alpha & Beta Diversity] Denoised sample not found")

        processSample(
            index,
            sample,
            metadataSample,
            denoisedSample,
            run,
            outputDataset,
            outputDir
        )


if __name__ == "__main__":
    initializeJob(main)
