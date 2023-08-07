from typing import List
from pathlib import Path
from zipfile import ZipFile

import csv
import logging

from coretex import CustomDataset, CustomSample, Experiment, folder_manager
from coretex.bioinformatics import ctx_qiime2

from .utils import summarizeSample, loadPairedEnd


def importSample(manifestPath: Path, sequenceType: str, inputFormat: str, outputDir: Path) -> Path:
    demuxPath = outputDir / "demux.qza"

    ctx_qiime2.toolsImport(
        sequenceType,
        str(manifestPath),
        str(demuxPath),
        inputFormat
    )

    demuxZipPath = outputDir / "demux.zip"
    with ZipFile(demuxZipPath, "w") as demuxFile:
        demuxFile.write(demuxPath, "demux.qza")

    return demuxZipPath


def importMetadata(metadata: CustomSample, outputDir: Path) -> Path:
    metadataZipPath = outputDir / "metadata.zip"
    with ZipFile(metadataZipPath, "w") as metadataFile:
        metadataFile.write(Path(metadata.path) / "metadata.tsv", "metadata.tsv")

    return metadataZipPath


def createManifestSingle(samples: List[CustomSample], manifestPath: Path) -> Path:
    with manifestPath.open("w") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "absolute-filepath"])

    for sample in samples:
        for fastqPath in Path(sample.path).iterdir():
            if fastqPath.suffix != ".fastq":
                continue

            sampleId = fastqPath.stem
            with manifestPath.open("a") as manifestFile:
                csv.writer(manifestFile, delimiter = "\t").writerow([sampleId, fastqPath])

    return manifestPath


def createManifestPaired(samples: List[CustomSample], manifestPath: Path) -> Path:
    with manifestPath.open("w") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"])

    for sample in samples:
        forwardPath, reversePath, sampleId = loadPairedEnd(sample)
        with manifestPath.open("a") as manifestFile:
            csv.writer(manifestFile, delimiter = "\t").writerow([sampleId, forwardPath, reversePath])

    return manifestPath


def importDemultiplexedSamples(
    dataset: CustomDataset,
    experiment: Experiment,
    pairedEnd: bool
) -> CustomDataset:

    logging.info(">> [Microbiome analysis] Preparing data for import into QIIME2 format")
    metadata = ctx_qiime2.getMetadataSample(experiment.dataset)

    outputDir = folder_manager.createTempFolder("import_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 1: Import",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Microbiome analysis] Failed to create output dataset")

    manifestPath = outputDir / "manifest.tsv"

    fastqSamples = ctx_qiime2.getFastqDPSamples(dataset)
    createManifestPaired(fastqSamples, manifestPath) if pairedEnd else createManifestSingle(fastqSamples, manifestPath)

    if pairedEnd:
        createManifestPaired(fastqSamples, manifestPath)
        sequenceType = "SampleData[PairedEndSequencesWithQuality]"
        inputFormat = "PairedEndFastqManifestPhred33V2"
    else:
        createManifestSingle(fastqSamples, manifestPath)
        sequenceType = "SampleData[SequencesWithQuality]"
        inputFormat = "SingleEndFastqManifestPhred33V2"

    logging.info(">> [Microbiome analysis] Importing data...")
    demuxZipPath = importSample(manifestPath, sequenceType, inputFormat, outputDir)
    demuxSample = ctx_qiime2.createSample("0-demux", outputDataset.id, demuxZipPath, experiment, "Step 1: Demultiplexing")

    metadataZipPath = importMetadata(metadata, outputDir)
    ctx_qiime2.createSample("0-import", outputDataset.id, metadataZipPath, experiment, "Step 1: Demultiplexing")

    demuxSample.download()
    demuxSample.unzip()

    logging.info(">> [Microbiome analysis] Creating summarization...")
    visualizationPath = summarizeSample(demuxSample, outputDir)
    ctx_qiime2.createSample("0-summary", outputDataset.id, visualizationPath, experiment, "Step 1: Demultiplexing")

    outputDataset.refresh()
    return outputDataset
