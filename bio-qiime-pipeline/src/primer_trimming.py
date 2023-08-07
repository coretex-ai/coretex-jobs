from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import Experiment, CustomDataset, CustomSample, folder_manager
from coretex.bioinformatics import cutadaptTrim

from .utils import loadPairedEnd


def forwardMetadata(sample: CustomSample, outputDataset: CustomDataset) -> None:
    sample.unzip()
    metadataZip = folder_manager.temp / "metadata.zip"
    with ZipFile(metadataZip, 'w', ZIP_DEFLATED) as archive:
        for path in sample.path.iterdir():
            archive.write(path, path.name)

    if CustomSample.createCustomSample(sample.name, outputDataset.id, metadataZip) is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to forward metadata to the output dataset")


def loadSingleEnd(sample: CustomSample) -> Path:
    sample.unzip()
    for filePath in sample.path.iterdir():
        if filePath.suffix != ".fastq":
            continue

        return filePath

    raise ValueError(f">> [Microbiome analysis] Sample \"{sample.name}\" does not contain fastq files")


def uploadSingle(file: Path, dataset: CustomDataset):
    zipPath = folder_manager.temp / f"{file.name}.zip"
    with ZipFile(zipPath, 'w', ZIP_DEFLATED) as archive:
        archive.write(file, file.name)

    if CustomSample.createCustomSample(file.name, dataset.id, zipPath) is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to upload trimmed reads")


def uploadPaired(forwardFile: Path, reverseFile: Path, sampleName: str, dataset: CustomDataset):
    zipPath = folder_manager.temp / f"{sampleName}.zip"
    with ZipFile(zipPath, 'w', ZIP_DEFLATED) as archive:
        archive.write(forwardFile, forwardFile.name)
        archive.write(reverseFile, reverseFile.name)

    if CustomSample.createCustomSample(sampleName, dataset.id, zipPath) is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to upload trimmed reads")


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

    logging.info(">> [Microbiome analysis] Loading files")
    if not pairedEnd:
        for sample in dataset.samples:
            if sample.name.startswith("metadata"):
                forwardMetadata(sample, outputDataset)
                continue

            inputFile = loadSingleEnd(sample)
            logging.info(f">> [Microbiome analysis] Trimming adapter sequences for {inputFile.name}")

            outputFile = forwardReadsFolder / inputFile.name
            cutadaptTrim(str(inputFile), str(outputFile), forwardAdapter)
            uploadSingle(outputFile, outputDataset)
    else:
        for sample in dataset.samples:
            if sample.name.startswith("metadata"):
                forwardMetadata(sample, outputDataset)
                continue

            forwardFile, reverseFile, sampleName = loadPairedEnd(sample)
            logging.info(f">> [Microbiome analysis] Trimming adapter sequences for {forwardFile.name} and {reverseFile.name}")

            forwardOutput = forwardReadsFolder / forwardFile.name
            reverseOutput = reverseReadsFolder / reverseFile.name
            cutadaptTrim(str(forwardFile), str(forwardOutput), forwardAdapter, str(reverseFile), str(reverseOutput), reverseAdapter)
            uploadPaired(forwardFile, reverseFile, sampleName, outputDataset)

    return CustomDataset.fetchById(outputDataset.id)
