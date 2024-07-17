from typing import Optional
from pathlib import Path
from zipfile import ZipFile

import os
import csv
import logging

from coretex import CustomDataset, CustomSample, TaskRun, folder_manager, currentTaskRun, createDataset
from coretex.bioinformatics import ctx_qiime2
from coretex.utils import CommandException


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
    outputDir: Path,
    threads: Optional[int]
) -> Path:

    phylogenyPath = sample.joinPath("rooted-tree.qza")
    outputPath = outputDir / "core-metrics-results"

    ctx_qiime2.diversityCoreMetricsPhylogenetic(
        str(phylogenyPath),
        str(tablePath),
        samplingDepth,
        str(metadataPath),
        str(outputPath),
        threads
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
    outputDataset: CustomDataset,
    outputPath: Path,
    taskRun: TaskRun
) -> None:

    ctx_qiime2.diversityAlphaGroupSignificance(
        str(alphaDiversityPath),
        str(metadataPath),
        str(outputPath)
    )

    ctx_qiime2.createSample(
        f"{sampleIndex}-{outputPath.stem}",
        outputDataset,
        outputPath,
        taskRun,
        "Step 7: Alpha & Beta diversity analysis"
    )


def diversityBetaGroupSignificance(
    distanceMatrixPath: Path,
    metadataPath: Path,
    metadataColumn: str,
    sampleIndex: int,
    outputDataset: CustomDataset,
    outputPath: Path,
    taskRun: TaskRun
) -> None:

    ctx_qiime2.diversityBetaGroupSignificance(
        str(distanceMatrixPath),
        str(metadataPath),
        metadataColumn,
        str(outputPath),
        True
    )

    ctx_qiime2.createSample(
        f"{sampleIndex}-{outputPath.stem}",
        outputDataset,
        outputPath,
        taskRun,
        "Step 7: Alpha & Beta diversity analysis"
    )


def emperorPlot(
    pcoaPath: Path,
    metadataPath: Path,
    sampleIndex: int,
    outputDataset: CustomDataset,
    outputPath: Path,
    taskRun: TaskRun
) -> None:

    ctx_qiime2.emperorPlot(
        str(pcoaPath),
        str(metadataPath),
        "days-since-experiment-start",
        str(outputPath)
    )

    ctx_qiime2.createSample(
        f"{sampleIndex}-{outputPath.stem}",
        outputDataset,
        outputPath,
        taskRun,
        "Step 7: Alpha & Beta diversity analysis"
    )


def diversityAlphaRarefaction(
    tablePath: Path,
    phylogenyPath: Path,
    maxDepth: int,
    metadataPath: Path,
    sampleIndex: int,
    outputDataset: CustomDataset,
    outputPath: Path,
    taskRun: TaskRun
) -> None:
    ctx_qiime2.diversityAlphaRarefaction(
        str(tablePath),
        str(phylogenyPath),
        maxDepth,
        str(metadataPath),
        str(outputPath)
    )

    ctx_qiime2.createSample(
        f"{sampleIndex}-{outputPath.stem}",
        outputDataset,
        outputPath,
        taskRun,
        "Step 7: Alpha & Beta diversity analysis"
    )


def processSample(
    index: int,
    sample: CustomSample,
    metadataSample: CustomSample,
    denoisedSample: CustomSample,
    taskRun: TaskRun,
    outputDataset: CustomDataset,
    outputDir: Path
) -> None:

    sample.unzip()
    metadataSample.unzip()
    denoisedSample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    metadataPath = ctx_qiime2.getMetadata(metadataSample)
    targetTypeColumn = taskRun.parameters["targetTypeColumn"].strip()

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
            taskRun.parameters["samplingDepth"],
            metadataPath,
            sampleOutputDir,
            taskRun.parameters["threads"]
        )

        coreMetricsSample = ctx_qiime2.createSample(
            f"{index}-core-metrics-phylogenetic",
            outputDataset,
            coreMetricsPath,
            taskRun,
            "Step 7: Alpha & Beta diversity analysis"
        )
    except CommandException:
        raise ValueError(">> [Qiime: Alpha & Beta Diversity] Failed to execute \"qiime diversity core-metrics-phylogenetic\"")

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
            outputDataset,
            sampleOutputDir / "faith-pd-group-significance.qzv",
            taskRun
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create faith_pd_vector.qza")

    logging.info(">> [Qiime: Alpha & Beta Diversity] Generating evenness_vector.qza")
    try:
        diversityAlphaGroupSignificance(
            coreMetricsSample.joinPath("evenness_vector.qza"),
            metadataPath,
            index,
            outputDataset,
            sampleOutputDir / "evenness-group-significance.qzv",
            taskRun
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
            outputDataset,
            sampleOutputDir / "unweighted-unifrac-body-site-significance.qzv",
            taskRun
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create unweighted_unifrac_distance_matrix.qza")

    try:
        diversityBetaGroupSignificance(
            coreMetricsSample.joinPath("unweighted_unifrac_distance_matrix.qza"),
            metadataPath,
            targetTypeColumn,
            index,
            outputDataset,
            sampleOutputDir / "unweighted-unifrac-subject-group-significance.qzv",
            taskRun
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
            outputDataset,
            sampleOutputDir / "unweighted-unifrac-emperor-days-since-experiment-start.qzv",
            taskRun
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create unweighted-unifrac-emperor-days-since-experiment-start.qzv")

    logging.info(">> [Qiime: Alpha & Beta Diversity] Generating bray_curtis_pcoa_results.qza")
    try:
        emperorPlot(
            coreMetricsSample.joinPath("bray_curtis_pcoa_results.qza"),
            metadataPath,
            index,
            outputDataset,
            sampleOutputDir / "bray-curtis-emperor-days-since-experiment-start.qzv",
            taskRun
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
            taskRun.parameters["maxDepth"],
            metadataPath,
            index,
            outputDataset,
            sampleOutputDir / "alpha-rarefaction.qzv",
            taskRun
        )
    except CommandException:
        logging.error(">> [Qiime: Alpha & Beta Diversity] Failed to create alpha-rarefaction.qzv")


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    # If GPU is detected but not configured properly we have
    # to disable its usage for unifrac otherwise experiment
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

    taskRun.dataset.download()

    phylogeneticTreeSamples = ctx_qiime2.getPhylogeneticTreeSamples(taskRun.dataset)
    if len(phylogeneticTreeSamples) == 0:
        raise ValueError(">> [Qiime: Alpha & Beta Diversity] Dataset has 0 phylogenetic tree samples")

    importedDataset: CustomDataset = taskRun.parameters["importedDataset"]
    importedDataset.download()

    denoisedDataset: CustomDataset = taskRun.parameters["denoisedDataset"]
    denoisedDataset.download()

    outputDir = folder_manager.createTempFolder("alpha_beta_output")

    datasetName = f"{taskRun.id}-step-7-alpha-and-beta-diversity"
    with createDataset(CustomDataset, datasetName, taskRun.projectId) as outputDataset:

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
                taskRun,
                outputDataset,
                outputDir
            )

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
