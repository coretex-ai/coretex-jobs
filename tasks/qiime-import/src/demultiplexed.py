from typing import List
from pathlib import Path
from zipfile import ZipFile

import csv
import logging

from coretex import SequenceDataset, CustomDataset, CustomSample, TaskRun, SequenceSample, createDataset
from coretex.bioinformatics import ctx_qiime2

from .utils import convertMetadata


def demuxSummarize(sample: CustomSample, outputDir: Path) -> Path:
    demuxPath = sample.path / "demux.qza"
    visualizationPath = outputDir / "demux.qzv"

    ctx_qiime2.demuxSummarize(str(demuxPath), str(visualizationPath))
    return visualizationPath


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
    taskRun: TaskRun,
    outputDir: Path
) -> None:

    outputDatasetName = f"{taskRun.id} - Step 1: Import - Demultiplexed"
    with createDataset(CustomDataset, outputDatasetName, taskRun.projectId) as outputDataset:

        logging.info(">> [Qiime: Import] Preparing demultiplexed data for import into Qiime2")
        inputPath = outputDir / "manifest.tsv"

        if dataset.isPairedEnd():
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
        demuxSample = ctx_qiime2.createSample("0-demux", outputDataset.id, importZipPath, taskRun, "Step 1: Import")

        metadataZipPath = importMetadata(dataset.metadata, outputDir, taskRun.parameters["metadataFileName"])
        ctx_qiime2.createSample("0-metadata", outputDataset.id, metadataZipPath, taskRun, "Step 1: Import")

        demuxSample.download()
        demuxSample.unzip()

        logging.info(">> [Qiime: Import] Creating summarization...")
        visualizationPath = demuxSummarize(demuxSample, outputDir)
        ctx_qiime2.createSample("0-summary", outputDataset.id, visualizationPath, taskRun, "Step 1: Import")

    taskRun.submitOutput("outputDataset", outputDataset)
