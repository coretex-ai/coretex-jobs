from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomDataset, CustomSample, Experiment, folder_manager
from coretex.project import initializeProject
from coretex.bioinformatics import ctx_qiime2


def deNovoClustering(sample: CustomSample, outputDir: Path, percentIdentity: float) -> None:
    tablePath = sample.path / "table.qza"
    sequencesPath = sample.path / "rep-seqs.qza"

    clusteredTablePath = outputDir / "clustered-table.qza"
    clusteredSequencesPath = outputDir / "clustered-seqs.qza"

    ctx_qiime2.vsearchClusterDeNovo(
        str(tablePath),
        str(sequencesPath),
        percentIdentity,
        str(clusteredTablePath),
        str(clusteredSequencesPath)
    )

    outputPath = outputDir / "otu.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(clusteredTablePath, clusteredTablePath.name)
        outputFile.write(clusteredSequencesPath, clusteredSequencesPath.name)

    return outputPath


def processSample(
    index: int,
    sample: CustomSample,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
) -> None:

    sample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    # Phylogenetic diversity analysis
    logging.info(">> [Microbiome analysis] Performing de novo clustering")
    percentIdentity = experiment.parameters["percentIdentity"]
    if percentIdentity <=0 or percentIdentity > 1:
        raise ValueError(">> [Qiime: Clustering] The percent identity parameter must be between 0 and 1.")

    otuPath = deNovoClustering(sample, sampleOutputDir, experiment.parameters["percentIdentity"])
    ctx_qiime2.createSample(f"{index}-otu-clusters", outputDataset.id, otuPath, experiment, "Step 4: OTU clustering")



def main(experiment: Experiment[CustomDataset]):
    dataset = experiment.dataset
    dataset.download()

    denoisedSamples = ctx_qiime2.getDenoisedSamples(dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Qiime: Clustering] Dataset has 0 denoised samples")

    outputDir = folder_manager.createTempFolder("otu_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 4: OTU clustering",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Qiime: Clustering] Failed to create output dataset")

    for sample in denoisedSamples:
        index = ctx_qiime2.sampleNumber(sample)
        processSample(index, sample, experiment, outputDataset, outputDir)




if __name__ == "__main__":
    initializeProject(main)
