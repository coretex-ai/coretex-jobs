from typing import List, Tuple, Optional
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import Experiment, CustomDataset, CustomSample, folder_manager
from coretex.project import initializeProject
from coretex.bioinformatics import cutadaptTrim, isPairedEnd


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


def loadPairedEnd(sample: CustomSample) -> Tuple[Path, Path, str]:
    sample.unzip()

    forwardPathList = list(sample.path.glob("*_R1_*.fastq"))
    reversePathList = list(sample.path.glob("*_R2_*.fastq"))

    if len(forwardPathList) > 0 and len(reversePathList) > 0:
        forwardPath = forwardPathList[0]
        reversePath = reversePathList[0]
    else:
        raise ValueError(f">> [Microbiome analysis] \"_R1_\" and \"_R2_\" not found, invalid paired-end sample: {sample.name}")

    return forwardPath, reversePath, forwardPath.name.split("_")[0]


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


def main(experiment: Experiment[CustomDataset]) -> None:
    forwardAdapter = experiment.parameters["forwardAdapter"]
    reverseAdapter = experiment.parameters["reverseAdapter"]

    dataset = experiment.dataset
    pairedEnd = isPairedEnd(dataset)

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


if __name__ == "__main__":
    initializeProject(main)
