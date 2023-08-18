from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomSample, CustomDataset, Experiment, folder_manager
from coretex.project import initializeProject
from coretex.bioinformatics import ctx_qiime2


def handleMetadata(sample: CustomSample, outputDataset: CustomDataset) -> Path:
    if CustomSample.createCustomSample(sample.name, outputDataset.id, sample.zipPath) is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to forward metadata to the output dataset")

    return list(sample.path.glob("*.tsv"))[0]


def summarizeSample(sample: CustomSample, outputDir: Path) -> Path:
    demuxPath = Path(sample.path) / "demux.qza"
    visualizationPath = outputDir / "demux.qzv"

    ctx_qiime2.demuxSummarize(str(demuxPath), str(visualizationPath))
    return visualizationPath


def demuxEmpSingleSample(sample: CustomSample, barcodesPath: Path, barcodeColumn: str, outputDir: Path) -> Path:
    samplePath = Path(sample.path)

    sequencesPath = samplePath / "multiplexedSequences.qza"

    demuxFilePath = outputDir / "demux.qza"
    demuxDetailsFilePath = outputDir / "demux-details.qza"

    ctx_qiime2.demuxEmpSingle(
        str(sequencesPath),
        str(barcodesPath),
        barcodeColumn,
        str(demuxFilePath),
        str(demuxDetailsFilePath)
    )

    demuxOutputPath = outputDir / "demux-output.zip"

    with ZipFile(demuxOutputPath, "w") as demuxOutput:
        demuxOutput.write(demuxFilePath, demuxFilePath.name)
        demuxOutput.write(demuxDetailsFilePath, demuxDetailsFilePath.name)

    return demuxOutputPath


def main(experiment: Experiment[CustomDataset]):
    dataset = experiment.dataset
    dataset.download()

    importedSamples = ctx_qiime2.getImportedSamples(dataset)
    if len(importedSamples) == 0:
        raise ValueError(">> [Qiime Demux] Dataset has 0 imported samples")

    outputDir = folder_manager.createTempFolder("demux_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 2: Demultiplexing",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Workspace] Failed to create output dataset")

    for sample in importedSamples:
        sample.unzip()

        index = ctx_qiime2.sampleNumber(sample)

        metadataSample = dataset.getSample(f"{index}-metadata")
        if metadataSample is None:
            raise ValueError(f">> [Microbiome analysis] Metadata sample not found")

        metadataPath = handleMetadata(metadataSample, outputDataset)
        demuxPath = demuxEmpSingleSample(sample, metadataPath, experiment.parameters["barcodeColumn"], outputDir)
        demuxSample = ctx_qiime2.createSample(f"{index}-demux", outputDataset.id, demuxPath, experiment, "Step 1: Demultiplexing")

        demuxSample.download()
        demuxSample.unzip()

        logging.info(">> [Microbiome analysis] Creating summarization")
        visualizationPath = summarizeSample(demuxSample, outputDir)
        ctx_qiime2.createSample(f"{index}-summary", outputDataset.id, visualizationPath, experiment, "Step 1: Demultiplexing")


if __name__ == "__main__":
    initializeProject(main)
