from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomSample, CustomDataset, Experiment, folder_manager
from coretex.bioinformatics import ctx_qiime2

from src.utils import demuxSummarize, isPairedEnd


def handleMetadata(sample: CustomSample, index: int,  outputDatasetId: int, experiment: Experiment) -> Path:
    ctx_qiime2.createSample(f"{index}-metadata", outputDatasetId, sample.zipPath, experiment, "Step 2: Demultiplexing")
    return ctx_qiime2.getMetadata(sample)


def demuxEmpSample(sample: CustomSample, barcodesPath: Path, barcodeColumn: str, outputDir: Path, pairedEnd: bool) -> Path:
    samplePath = Path(sample.path)

    sequencesPath = samplePath / "multiplexed-sequences.qza"

    demuxFilePath = outputDir / "demux.qza"
    demuxDetailsFilePath = outputDir / "demux-details.qza"

    if not pairedEnd:
        ctx_qiime2.demuxEmpSingle(
            str(sequencesPath),
            str(barcodesPath),
            barcodeColumn,
            str(demuxFilePath),
            str(demuxDetailsFilePath)
        )
    else:
        ctx_qiime2.demuxEmpPaired(
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


def demultiplexing(
    dataset: CustomDataset,
    experiment: Experiment
) -> CustomDataset:

    importedSamples = ctx_qiime2.getImportedSamples(dataset)
    if len(importedSamples) == 0:
        raise ValueError(">> [Qiime: Demux] Dataset has 0 imported samples")

    outputDir = folder_manager.createTempFolder("demux_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 2: Demultiplexing",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Qiime: Demux] Failed to create output dataset")

    for sample in importedSamples:
        sample.unzip()

        index = ctx_qiime2.sampleNumber(sample)

        metadataSample = dataset.getSample(f"{index}-metadata")
        if metadataSample is None:
            raise ValueError(f">> [Qiime: Demux] Metadata sample not found")

        metadataPath = handleMetadata(metadataSample, index, outputDataset.id, experiment)
        demuxPath = demuxEmpSample(
            sample,
            metadataPath,
            experiment.parameters["barcodeColumn"],
            outputDir,
            isPairedEnd(sample)
        )

        demuxSample = ctx_qiime2.createSample(f"{index}-demux", outputDataset.id, demuxPath, experiment, "Step 2: Demultiplexing")

        demuxSample.download()
        demuxSample.unzip()

        logging.info(">> [Qiime: Demux] Creating summarization")
        visualizationPath = demuxSummarize(demuxSample, outputDir)
        ctx_qiime2.createSample(f"{index}-summary", outputDataset.id, visualizationPath, experiment, "Step 2: Demultiplexing")

    outputDataset.refresh()
    return outputDataset
