from typing import List
from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomDataset, CustomSample, Experiment, folder_manager
from coretex.bioinformatics import ctx_qiime2

from .utils import summarizeSample


def importSample(sequencesPath: Path, metadataPath: Path, sequenceType: str, outputDir: Path) -> Path:
    importedSequencesPath = outputDir / "sequences.qza"

    ctx_qiime2.toolsImport(sequenceType, str(sequencesPath), str(importedSequencesPath))

    outputPath = outputDir / "import-output.zip"

    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(metadataPath, metadataPath.name)
        outputFile.write(importedSequencesPath, importedSequencesPath.name)

    return outputPath


def demuxEmpSingleSample(sample: CustomSample, barcodesPath: Path, barcodeColumn: str, outputDir: Path) -> Path:
    samplePath = Path(sample.path)

    sequencesPath = samplePath / "sequences.qza"

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


def processSample(
    sequenceFolderPath: Path,
    metadataPath: Path,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
):
    # experiment.createQiimeArtifact(f"Step 1: Demultiplexing/{index}-original", Path(sample.zipPath))

    sequencesPath = sequenceFolderPath / "sequences.fastq.gz"
    barcodesPath = sequenceFolderPath / "barcodes.fastq.gz"

    if not sequencesPath.exists():
        source = sequenceFolderPath / "sequences.fastq"
        ctx_qiime2.compressGzip(source, sequencesPath)

    if not barcodesPath.exists():
        source = sequenceFolderPath / "barcodes.fastq"
        ctx_qiime2.compressGzip(source, barcodesPath)

    # First step:
    # Importing sample using qiime2 to generate qza file from fastq/fasta sequences
    # TODO:
    # Don't save
    # If needed keep comment
    # Local sample
    # Separate function
    logging.info(">> [Microbiome analysis] Importing sample")
    importedFilePath = importSample(sequenceFolderPath, metadataPath, experiment.parameters["sequenceType"], outputDir)
    importedSample = ctx_qiime2.createSample("0-import", outputDataset.id, importedFilePath, experiment, "Step 1: Demultiplexing")

    # Second step:
    # Demultiplexing sequences to extract all barcode occurences from imported qza files
    importedSample.download()
    importedSample.unzip()

    logging.info(">> [Microbiome analysis] Demultiplexing sample")
    demuxPath = demuxEmpSingleSample(importedSample, metadataPath, experiment.parameters["barcodeColumn"], outputDir)
    demuxSample = ctx_qiime2.createSample("0-demux", outputDataset.id, demuxPath, experiment, "Step 1: Demultiplexing")

    # Third step:
    # Summarize demultiplexed sequences to visualize the results
    demuxSample.download()
    demuxSample.unzip()

    logging.info(">> [Microbiome analysis] Creating summarization")
    visualizationPath = summarizeSample(demuxSample, outputDir)
    ctx_qiime2.createSample("0-summary", outputDataset.id, visualizationPath, experiment, "Step 1: Demultiplexing")


def demultiplexing(
    dataset: CustomDataset,
    experiment: Experiment[CustomDataset],
    outputDataset: CustomDataset,
    outputDir: Path
):

    logging.info(">> [Microbiome analysis] Multiplexed samples detected. Procceeding with demultiplexing")

    dataDir = folder_manager.createTempFolder("data_dir")
    sequencesPath = dataDir / "sequences.fastq"
    barcodesPath = dataDir / "barcodes.fastq"

    fastqSamples = ctx_qiime2.getFastqMPSamples(dataset)
    for sample in fastqSamples:
        sample.unzip()

        samplePath = Path(sample.path)
        sampleSequences = samplePath / "sequences" / "sequences.fastq"
        sampleBarcodes = samplePath / "sequences" / "barcodes.fastq"

        with sequencesPath.open("a") as sequencesFile:
            sequencesFile.write(sampleSequences.read_text())

        with barcodesPath.open("a") as barcodesFile:
            barcodesFile.write(sampleBarcodes.read_text())

    metadataPath = Path(experiment.dataset.samples[0].path) / experiment.parameters["barcodesFileName"]
    processSample(dataDir, metadataPath, experiment, outputDataset, outputDir)
