from typing import Optional
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import Experiment, CustomSample, folder_manager, SequenceDataset, SequenceSample, currentExperiment
from coretex.bioinformatics import cutadaptTrim


def uploadTrimmedReads(sampleName: str, dataset: SequenceDataset, forwardFile: Path, reverseFile: Optional[Path] = None):
    zipPath = folder_manager.temp / f"{sampleName}.zip"
    with ZipFile(zipPath, 'w', ZIP_DEFLATED) as archive:
        archive.write(forwardFile, forwardFile.name)
        if reverseFile:
            archive.write(reverseFile, reverseFile.name)

    if CustomSample.createCustomSample(sampleName, dataset.id, zipPath) is None:
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


def main() -> None:
    experiment: Experiment[SequenceDataset] = currentExperiment()
    experiment.setDatasetType(SequenceDataset)

    forwardAdapter = experiment.parameters["forwardAdapter"]
    reverseAdapter = experiment.parameters["reverseAdapter"]

    dataset = experiment.dataset
    dataset.download()

    pairedEnd = dataset.isPairedEnd()

    if forwardAdapter is None:
        forwardAdapter = "X"

    if reverseAdapter is None:
        reverseAdapter = "X"

    forwardReadsFolder = folder_manager.createTempFolder("forwardReads")
    if pairedEnd:
        reverseReadsFolder = folder_manager.createTempFolder("revereseReads")

    outputDataset = SequenceDataset.createSequenceDataset(
        f"{experiment.id} - Cutadapt Output",
        experiment.spaceId,
        dataset.metadata.zipPath
    )

    if outputDataset is None:
        raise RuntimeError(">> [Microbiome analysis] Failed to create coretex dataset")

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


if __name__ == "__main__":
    main()
