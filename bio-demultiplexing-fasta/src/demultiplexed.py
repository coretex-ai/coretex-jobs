from typing import List, Tuple
from pathlib import Path
from zipfile import ZipFile

import csv
import logging

from coretex import SequenceDataset, CustomDataset, CustomSample, Experiment, SequenceSample
from coretex.bioinformatics import ctx_qiime2

from .utils import summarizeSample


def importSample(inputPath: Path, sequenceType: str, inputFormat: str, outputDir: Path) -> Path:
    demuxPath = outputDir / "demux.qza"

    ctx_qiime2.toolsImport(
        sequenceType,
        str(inputPath),
        str(demuxPath),
        inputFormat
    )

    demuxZipPath = outputDir / "demux.zip"
    with ZipFile(demuxZipPath, "w") as demuxFile:
        demuxFile.write(demuxPath, "demux.qza")

    return demuxZipPath


def importMetadata(metadata: CustomSample, outputDir: Path) -> Path:
    metadata.unzip()
    metadataZipPath = outputDir / "metadata.zip"

    with ZipFile(metadataZipPath, "w") as metadataFile:
        metadataFile.write(Path(metadata.path) / "metadata.tsv", "metadata.tsv")

    return metadataZipPath


def createManifestSingle(samples: List[SequenceSample], manifestPath: Path) -> Path:
    with manifestPath.open("a") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "absolute-filepath"])

        for sample in samples:
            sample.unzip()

            csv.writer(manifestFile, delimiter = "\t").writerow([sample.sequencePath.stem, sample.sequencePath])

    return manifestPath


def createManifestPaired(samples: List[SequenceSample], manifestPath: Path) -> Path:
    with manifestPath.open("a") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"])

        for sample in samples:
            sample.unzip()

            forwardPath = sample.forwardPath
            reversePath = sample.reversePath
            csv.writer(manifestFile, delimiter = "\t").writerow([forwardPath.name.split("_")[0], forwardPath, reversePath])

    return manifestPath


def importDemultiplexedSamples(
    dataset: SequenceDataset,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
):

    logging.info(">> [Microbiome analysis] Preparing data for import into QIIME2 format")
    inputPath = outputDir / "manifest.tsv"

    if dataset.isPairedEnd():
        createManifestPaired(dataset.samples, inputPath)
        sequenceType = "SampleData[PairedEndSequencesWithQuality]"
        inputFormat = "PairedEndFastqManifestPhred33V2"
    else:
        createManifestSingle(dataset.samples, inputPath)
        sequenceType = "SampleData[SequencesWithQuality]"
        inputFormat = "SingleEndFastqManifestPhred33V2"

    logging.info(">> [Microbiome analysis] Importing data...")
    demuxZipPath = importSample(inputPath, sequenceType, inputFormat, outputDir)
    demuxSample = ctx_qiime2.createSample("0-demux", outputDataset.id, demuxZipPath, experiment, "Step 1: Demultiplexing")

    metadataZipPath = importMetadata(dataset.metadata, outputDir)
    ctx_qiime2.createSample("0-import", outputDataset.id, metadataZipPath, experiment, "Step 1: Demultiplexing")

    demuxSample.download()
    demuxSample.unzip()

    logging.info(">> [Microbiome analysis] Creating summarization...")
    visualizationPath = summarizeSample(demuxSample, outputDir)
    ctx_qiime2.createSample("0-summary", outputDataset.id, visualizationPath, experiment, "Step 1: Demultiplexing")
