from typing import Optional
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import Run, SequenceDataset, CustomSample, SequenceSample, folder_manager
from coretex.bioinformatics import cutadaptTrim


def forwardMetadata(sample: CustomSample, outputDataset: SequenceDataset) -> None:
    sample.unzip()

    metadataZip = folder_manager.temp / "_metadata.zip"
    with ZipFile(metadataZip, 'w', ZIP_DEFLATED) as archive:
        for path in sample.path.iterdir():
            archive.write(path, path.name)

    if CustomSample.createCustomSample("_metadata", outputDataset.id, metadataZip) is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to forward metadata to the output dataset")


def uploadTrimmedReads(sampleName: str, dataset: SequenceDataset, forwardFile: Path, reverseFile: Optional[Path] = None):
    zipPath = folder_manager.temp / f"{sampleName}.zip"
    with ZipFile(zipPath, 'w', ZIP_DEFLATED) as archive:
        archive.write(forwardFile, forwardFile.name)
        if reverseFile:
            archive.write(reverseFile, reverseFile.name)

    if SequenceSample.createSequenceSample(zipPath, dataset.id) is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to upload trimmed reads")


def trimSingleEnd(
    sample: SequenceSample,
    forwardAdapter: str,
    forwardReadsFolder: Path,
    outputDataset: SequenceDataset
) -> None:

    inputFile = sample.sequencePath
    logging.info(f">> [Microbiome analysis] Trimming adapter sequences for {inputFile.name}")

    outputFile = forwardReadsFolder / inputFile.name
    cutadaptTrim(str(inputFile), str(outputFile), forwardAdapter)
    uploadTrimmedReads(inputFile.name.split("_")[0], outputDataset, outputFile)


def trimPairedEnd(
    sample: SequenceSample,
    forwardAdapter: str,
    reverseAdapter: str,
    forwardReadsFolder: Path,
    reverseReadsFolder: Path,
    outputDataset: SequenceDataset
) -> None:

    forwardFile = sample.forwardPath
    reverseFile = sample.reversePath
    logging.info(f">> [Microbiome analysis] Trimming adapter sequences for {forwardFile.name} and {reverseFile.name}")

    forwardOutput = forwardReadsFolder / forwardFile.name
    reverseOutput = reverseReadsFolder / reverseFile.name
    cutadaptTrim(str(forwardFile), str(forwardOutput), forwardAdapter, str(reverseFile), str(reverseOutput), reverseAdapter)
    uploadTrimmedReads(forwardFile.name.split("_")[0], outputDataset, forwardFile, reverseFile)


def primerTrimming(dataset: SequenceDataset, run: Run, pairedEnd: bool) -> SequenceDataset:
    forwardAdapter = run.parameters["forwardAdapter"]
    reverseAdapter = run.parameters["reverseAdapter"]

    # In case no adapter in entered, "X" will act as placeholder as no
    # sequence should start with the letter X
    if forwardAdapter is None:
        forwardAdapter = "X"

    if reverseAdapter is None:
        reverseAdapter = "X"

    forwardReadsFolder = folder_manager.createTempFolder("forwardReads")
    if pairedEnd:
        reverseReadsFolder = folder_manager.createTempFolder("revereseReads")

    outputDataset = SequenceDataset.createDataset(f"{run.id} - Cutadapt Output", run.spaceId)
    if outputDataset is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to create coretex dataset")

    forwardMetadata(dataset.metadata, outputDataset)
    for sample in dataset.samples:
        sample.unzip()

        if not pairedEnd:
            trimSingleEnd(
                sample,
                forwardAdapter,
                forwardReadsFolder,
                outputDataset
            )
        else:
            trimPairedEnd(
                sample,
                forwardAdapter,
                reverseAdapter,
                forwardReadsFolder,
                reverseReadsFolder,
                outputDataset
            )

    outputDataset.refresh()
    return outputDataset
