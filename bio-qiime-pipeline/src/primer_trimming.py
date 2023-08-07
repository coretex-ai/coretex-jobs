from typing import List, Optional
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import Experiment, CustomDataset, CustomSample, folder_manager
from coretex.bioinformatics import cutadaptTrim

from .utils import loadPairedEnd


def forwardMetadata(sample: CustomSample, outputDataset: CustomDataset) -> None:
    sample.unzip()

    metadataZip = folder_manager.temp / "_metadata.zip"
    with ZipFile(metadataZip, 'w', ZIP_DEFLATED) as archive:
        for path in sample.path.iterdir():
            archive.write(path, path.name)

    if CustomSample.createCustomSample("_metadata", outputDataset.id, metadataZip) is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to forward metadata to the output dataset")


def loadSingleEnd(sample: CustomSample) -> Path:
    sample.unzip()

    for filePath in sample.path.iterdir():
        if filePath.suffix != ".fastq":
            continue

        return filePath

    raise ValueError(f">> [Microbiome analysis] Sample \"{sample.name}\" does not contain fastq files")


def uploadTrimmedReads(sampleName: str, dataset: CustomDataset, forwardFile: Path, reverseFile: Optional[Path] = None):
    zipPath = folder_manager.temp / f"{sampleName}.zip"
    with ZipFile(zipPath, 'w', ZIP_DEFLATED) as archive:
        archive.write(forwardFile, forwardFile.name)
        if reverseFile:
            archive.write(reverseFile, reverseFile.name)

    if CustomSample.createCustomSample(sampleName, dataset.id, zipPath) is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to upload trimmed reads")


def trimSingleEnd(
    samples: List[CustomSample],
    forwardAdapter: str,
    forwardReadsFolder: Path,
    outputDataset: CustomDataset
) -> None:

    for sample in samples:
        if sample.name.startswith("_metadata"):
            forwardMetadata(sample, outputDataset)
            continue

        inputFile = loadSingleEnd(sample)
        logging.info(f">> [Microbiome analysis] Trimming adapter sequences for {inputFile.name}")

        outputFile = forwardReadsFolder / inputFile.name
        cutadaptTrim(str(inputFile), str(outputFile), forwardAdapter)
        uploadTrimmedReads(outputFile.stem, outputDataset, outputFile)


def trimPairedEnd(
    samples: List[CustomSample],
    forwardAdapter: str,
    reverseAdapter: str,
    forwardReadsFolder: Path,
    reverseReadsFolder: Path,
    outputDataset: CustomDataset
) -> None:

    for sample in samples:
        if sample.name.startswith("_metadata"):
            forwardMetadata(sample, outputDataset)
            continue

        forwardFile, reverseFile, sampleName = loadPairedEnd(sample)
        logging.info(f">> [Microbiome analysis] Trimming adapter sequences for {forwardFile.name} and {reverseFile.name}")

        forwardOutput = forwardReadsFolder / forwardFile.name
        reverseOutput = reverseReadsFolder / reverseFile.name
        cutadaptTrim(str(forwardFile), str(forwardOutput), forwardAdapter, str(reverseFile), str(reverseOutput), reverseAdapter)
        uploadTrimmedReads(sampleName, outputDataset, forwardFile, reverseFile)


def primerTrimming(dataset: CustomDataset, experiment: Experiment, pairedEnd: bool) -> CustomDataset:
    forwardAdapter = experiment.parameters["forwardAdapter"]
    reverseAdapter = experiment.parameters["reverseAdapter"]

    if forwardAdapter is None:
        forwardAdapter = ""

    if reverseAdapter is None:
        reverseAdapter = ""

    forwardReadsFolder = folder_manager.createTempFolder("forwardReads")
    if pairedEnd:
        reverseReadsFolder = folder_manager.createTempFolder("revereseReads")

    outputDataset = CustomDataset.createDataset(f"{experiment.id} - Cutadapt Output", experiment.spaceId)
    if outputDataset is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to create coretex dataset")

    if not pairedEnd:
        trimSingleEnd(
            dataset.samples,
            forwardAdapter,
            forwardReadsFolder,
            outputDataset
        )
    else:
        trimPairedEnd(
            dataset.samples,
            forwardAdapter,
            reverseAdapter,
            forwardReadsFolder,
            reverseReadsFolder,
            outputDataset
        )

    outputDataset.refresh()
    return outputDataset
