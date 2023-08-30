from typing import List
from pathlib import Path
from zipfile import ZipFile

import csv
import logging

from coretex import SequenceDataset, CustomDataset, CustomSample, Experiment, SequenceSample, folder_manager
from coretex.bioinformatics import ctx_qiime2

from .utils import convertMetadata, demuxSummarize
from .caching import getCacheNameOne


def importSample(inputPath: Path, sequenceType: str, inputFormat: str, outputDir: Path) -> Path:
    importedSequencesPath = outputDir / "demux.qza"

    ctx_qiime2.toolsImport(
        sequenceType,
        str(inputPath),
        str(importedSequencesPath),
        inputFormat
    )

    importZipPath = outputDir / "demux.zip"
    with ZipFile(importZipPath, "w") as importFile:
        importFile.write(importedSequencesPath, "demux.qza")

    return importZipPath


def importMetadata(metadata: CustomSample, outputDir: Path, metadataFileName: str) -> Path:
    metadata.unzip()

    metadataPath = convertMetadata(metadata.path / metadataFileName)
    metadataZipPath = outputDir / "metadata.zip"

    with ZipFile(metadataZipPath, "w") as metadataFile:
        metadataFile.write(metadataPath, metadataPath.name)

    return metadataZipPath


def createManifestSingle(samples: List[SequenceSample], manifestPath: Path) -> Path:
    with manifestPath.open("w") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "absolute-filepath"])

    for sample in samples:
        sample.unzip()
        fastqPath = sample.sequencePath

        with manifestPath.open("a") as manifestFile:
            csv.writer(manifestFile, delimiter = "\t").writerow([fastqPath.stem, fastqPath])

    return manifestPath


def createManifestPaired(samples: List[SequenceSample], manifestPath: Path) -> Path:
    with manifestPath.open("w") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"])

    for sample in samples:
        sample.unzip()
        forwardPath = sample.forwardPath
        reversePath = sample.reversePath

        with manifestPath.open("a") as manifestFile:
            csv.writer(manifestFile, delimiter = "\t").writerow([forwardPath.name.split("_")[0], forwardPath, reversePath])

    return manifestPath


def importDemultiplexed(
    dataset: SequenceDataset,
    experiment: Experiment,
    pairedEnd: bool
) -> CustomDataset:

    if pairedEnd:
        logging.info(">> [Qiime: Import] Demultiplexed paired-end reads detected")
    else:
        logging.info(">> [Qiime: Import] Demultiplexed single-end reads detected")

    logging.info(">> [Qiime: Import] Preparing data for import into QIIME2")
    outputDir = folder_manager.createTempFolder("import_output")
    outputDataset = CustomDataset.createDataset(
        getCacheNameOne(experiment),
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Qiime: Import] Failed to create output dataset")

    logging.info(">> [Qiime: Import] Preparing demultiplexed data for import into Qiime2")
    inputPath = outputDir / "manifest.tsv"

    if pairedEnd:
        createManifestPaired(dataset.samples, inputPath)
        sequenceType = "SampleData[PairedEndSequencesWithQuality]"
        inputFormat = "PairedEndFastqManifestPhred33V2"
    else:
        createManifestSingle(dataset.samples, inputPath)
        sequenceType = "SampleData[SequencesWithQuality]"
        inputFormat = "SingleEndFastqManifestPhred33V2"

    logging.info(">> [Qiime: Import] Importing data...")
    importZipPath = importSample(inputPath, sequenceType, inputFormat, outputDir)

    logging.info(">> [Qiime: Import] Uploading sample")
    demuxSample = ctx_qiime2.createSample("0-demux", outputDataset.id, importZipPath, experiment, "Step 1: Import")

    metadataZipPath = importMetadata(dataset.metadata, outputDir, experiment.parameters["metadataFileName"])
    ctx_qiime2.createSample("0-metadata", outputDataset.id, metadataZipPath, experiment, "Step 1: Import")

    demuxSample.download()
    demuxSample.unzip()

    logging.info(">> [Qiime: Import] Creating summarization...")
    visualizationPath = demuxSummarize(demuxSample, outputDir)
    ctx_qiime2.createSample("0-summary", outputDataset.id, visualizationPath, experiment, "Step 1: Import")

    outputDataset.refresh()
    return outputDataset
