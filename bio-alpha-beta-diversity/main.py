from pathlib import Path
from zipfile import ZipFile

import os
import logging

from coretex import CustomDataset, CustomSample, Experiment
from coretex.project import initializeProject
from coretex.folder_management import FolderManager
from coretex.bioinformatics import qiime2 as ctx_qiime2


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
    experiment: Experiment
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
        experiment,
        "Step 4: Alpha & Beta diversity analysis"
    )


def diversityBetaGroupSignificance(
    distanceMatrixPath: Path,
    metadataPath: Path,
    metadataColumn: str,
    sampleIndex: int,
    outputDatasetId: int,
    outputPath: Path,
    experiment: Experiment
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
        experiment,
        "Step 4: Alpha & Beta diversity analysis"
    )


def emperorPlot(
    pcoaPath: Path,
    metadataPath: Path,
    sampleIndex: int,
    outputDatasetId: int,
    outputPath: Path,
    experiment: Experiment
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
        experiment,
        "Step 4: Alpha & Beta diversity analysis"
    )


def diversityAlphaRarefaction(
    tablePath: Path,
    phylogenyPath: Path,
    maxDepth: int,
    metadataPath: Path,
    sampleIndex: int,
    outputDatasetId: int,
    outputPath: Path,
    experiment: Experiment
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
        experiment,
        "Step 4: Alpha & Beta diversity analysis"
    )


def processSample(
    index: int,
    sample: CustomSample,
    importedSample: CustomSample,
    denoisedSample: CustomSample,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
):

    sample.unzip()
    importedSample.unzip()
    denoisedSample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    metadataPath = importedSample.joinPath(experiment.parameters["barcodesFileName"])

    # First step:
    # Apply the core-metrics-phylogenetic method, which rarefies a
    # FeatureTable[Frequency] to a user-specified depth, computes
    # several alpha and beta diversity metrics, and generates principle
    # coordinates analysis (PCoA) plots using Emperor for each
    # of the beta diversity metrics.
    coreMetricsPath = diversityCoreMetricsPhylogeneticSample(
        sample,
        denoisedSample.joinPath("table.qza"),
        experiment.parameters["samplingDepth"],
        metadataPath,
        sampleOutputDir
    )

    coreMetricsSample = ctx_qiime2.createSample(
        f"{index}-core-metrics-phylogenetic",
        outputDataset.id,
        coreMetricsPath,
        experiment,
        "Step 4: Alpha & Beta diversity analysis"
    )

    # Second step:
    # Explore the microbial composition of the samples in the context of the sample metadata
    coreMetricsSample.download()
    coreMetricsSample.unzip()

    try:
        diversityAlphaGroupSignificance(
            coreMetricsSample.joinPath("faith_pd_vector.qza"),
            metadataPath,
            index,
            outputDataset.id,
            sampleOutputDir / "faith-pd-group-significance.qzv",
            experiment
        )
    except ctx_qiime2.QiimeCommandException:
        logging.error(">> [Microbiome analysis] Failed to create faith_pd_vector.qza")

    try:
        diversityAlphaGroupSignificance(
            coreMetricsSample.joinPath("evenness_vector.qza"),
            metadataPath,
            index,
            outputDataset.id,
            sampleOutputDir / "evenness-group-significance.qzv",
            experiment
        )
    except ctx_qiime2.QiimeCommandException:
        logging.error(">> [Microbiome analysis] Failed to create evenness_vector.qza")

    # Third step:
    # Analyze sample composition in the context of categorical metadata using PERMANOVA
    # Test whether distances between samples within a group, such as
    # samples from the same body site (e.g., gut), are more similar to
    # each other then they are to samples from the other
    # groups (e.g., tongue, left palm, and right palm).
    try:
        diversityBetaGroupSignificance(
            coreMetricsSample.joinPath("unweighted_unifrac_distance_matrix.qza"),
            metadataPath,
            "body-site",
            index,
            outputDataset.id,
            sampleOutputDir / "unweighted-unifrac-body-site-significance.qzv",
            experiment
        )
    except ctx_qiime2.QiimeCommandException:
        logging.error(">> [Microbiome analysis] Failed to create unweighted_unifrac_distance_matrix.qza")

    try:
        diversityBetaGroupSignificance(
            coreMetricsSample.joinPath("unweighted_unifrac_distance_matrix.qza"),
            metadataPath,
            "body-site",
            index,
            outputDataset.id,
            sampleOutputDir / "unweighted-unifrac-subject-group-significance.qzv",
            experiment
        )
    except ctx_qiime2.QiimeCommandException:
        logging.error(">> [Microbiome analysis] Failed to create unweighted_unifrac_distance_matrix.qza")

    # Fourth step:
    # Exploring microbial community composition in the context of sample metadata using ordination
    emperorPlot(
        coreMetricsSample.joinPath("unweighted_unifrac_pcoa_results.qza"),
        metadataPath,
        index,
        outputDataset.id,
        sampleOutputDir / "unweighted-unifrac-emperor-days-since-experiment-start.qzv",
        experiment
    )

    emperorPlot(
        coreMetricsSample.joinPath("bray_curtis_pcoa_results.qza"),
        metadataPath,
        index,
        outputDataset.id,
        sampleOutputDir / "bray-curtis-emperor-days-since-experiment-start.qzv",
        experiment
    )

    # Fifth step:
    # This visualizer computes one or more alpha diversity metrics at multiple sampling depths,
    # in steps between 1 and the value provided as --p-max-depth
    diversityAlphaRarefaction(
        denoisedSample.joinPath("table.qza"),
        sample.joinPath("rooted-tree.qza"),
        experiment.parameters["maxDepth"],
        metadataPath,
        index,
        outputDataset.id,
        sampleOutputDir / "alpha-rarefaction.qzv",
        experiment
    )


def main(experiment: Experiment[CustomDataset]):
    # If GPU is detected but not configured properly we have
    # to disable its usage for unifrac otherwise experiment
    # will crash
    try:
        from py3nvml import py3nvml
        py3nvml.nvmlInit()

        os.environ["UNIFRAC_USE_GPU"] = "Y"
        logging.info(">> [Microbiome analysis] GPU will be used for \"unifrac\" calculations")

        py3nvml.nvmlShutdown()
    except:
        os.environ["UNIFRAC_USE_GPU"] = "N"

        logging.warning(">> [Microbiome analysis] GPU will not be used for \"unifrac\" calculations")

    experiment.dataset.download()

    phylogeneticTreeSamples = ctx_qiime2.getPhylogeneticTreeSamples(experiment.dataset)
    if len(phylogeneticTreeSamples) == 0:
        raise ValueError(">> [Microbiome analysis] Dataset has 0 phylogenetic tree samples")

    importedDataset: CustomDataset = experiment.parameters["importedDataset"]
    importedDataset.download()

    denoisedDataset: CustomDataset = experiment.parameters["denoisedDataset"]
    denoisedDataset.download()

    outputDir = Path(FolderManager.instance().createTempFolder("qiime_output"))
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 4: Alpha & Beta diversity",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Microbiome analysis] Failed to create output dataset")

    for sample in phylogeneticTreeSamples:
        index = ctx_qiime2.sampleNumber(sample)

        importedSample = importedDataset.getSample(f"{index}-import")
        if importedSample is None:
            raise ValueError(f">> [Microbiome analysis] Imported sample not found")

        denoisedSample = denoisedDataset.getSample(f"{index}-denoise")
        if denoisedSample is None:
            raise ValueError(f">> [Microbiome analysis] Denoised sample not found")

        processSample(
            index,
            sample,
            importedSample,
            denoisedSample,
            experiment,
            outputDataset,
            outputDir
        )


if __name__ == "__main__":
    initializeProject(main)
