from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import Experiment, CustomDataset, CustomSample, folder_manager
from coretex.project import initializeProject
from coretex.bioinformatics import cutadaptTrim


def loadSingleEnd(dataset: CustomDataset) -> list[Path]:
    filePaths: list[Path] = []
    for sample in dataset.samples:
        sample.unzip()
        for filePath in sample.path.iterdir():
            if filePath.suffix == ".fasta" or filePath.suffix == ".fastq":
                filePaths.append(filePath)

    return filePaths


def loadPairedEnd(dataset: CustomDataset) -> tuple[list[Path], list[Path]]:
    forewardPaths: list[Path] = []
    reversePaths: list[Path] = []
    for sample in dataset.samples:
        sample.unzip()
        for filePath in sample.path.iterdir():
            if filePath.suffix == ".fasta" or filePath.suffix == ".fastq":
                if sample.name.startswith("R1"):
                    forewardPaths.append(filePath)
                elif sample.name.startswith("R2"):
                    reversePaths.append(filePath)
                else:
                    raise ValueError(">> [Primer Removal] For paired-end reads, the foreward reads must be in samples with \"R1\" at the start of their names, while the reverse reads must be in samples with \"R2\" in their names. There should be no samples names that don't start with \"ßR1\" or \"R2\"")

    return forewardPaths, reversePaths


def main(experiment: Experiment[CustomDataset]):
    forewardAdapter = experiment.parameters["forewardAdapter"]
    reverseAdapter = experiment.parameters["reverseAdapter"]

    pairedEnd = reverseAdapter is not None

    forewardReadsFolder = folder_manager.createTempFolder("forewardReads")
    if pairedEnd:
        reverseReadsFolder = folder_manager.createTempFolder("revereseReads")

    dataset = experiment.dataset
    dataset.download()

    logging.info(">> [Primer Removal] Loading files")
    if not pairedEnd:
        filePaths = loadSingleEnd(dataset)
        for inputFile in filePaths:
            logging.info(f">> [Primer Removal] Trimming adapter sequences for {inputFile.name}")

            outputFile = forewardReadsFolder / inputFile.name
            cutadaptTrim(str(inputFile), str(outputFile), forewardAdapter)
    else:
        forewardPaths, reversePaths = loadPairedEnd(dataset)
        for forewardFile, reverseFile in zip(forewardPaths, reversePaths):
            logging.info(f">> [Primer Removal] Trimming adapter sequences for {forewardFile.name} and {reverseFile.name}")

            forewardOutput = forewardReadsFolder / forewardFile.name
            reverseOutput = reverseReadsFolder / reverseFile.name
            cutadaptTrim(str(forewardFile), str(forewardOutput), forewardAdapter, str(reverseFile), str(reverseOutput), reverseAdapter)

    outputDataset = CustomDataset.createDataset(f"{experiment.id} - Cutadapt Output", experiment.spaceId)
    if outputDataset is None:
        raise RuntimeError(">> [Primer Removal] Failed to create coretex dataset")

    logging.info(f">> [Primer Removal] Trimming finished. Now uploading to coretex as dataset: {outputDataset.name}")

    forewardZip = folder_manager.temp / "forewardReads.zip"
    with ZipFile(forewardZip, 'w', ZIP_DEFLATED) as archive:
        for filePath in forewardReadsFolder.iterdir():
            archive.write(filePath,filePath.name)

    forewardSampleName = "R1_foreward_reads" if pairedEnd else "trimmed_reads"
    if CustomSample.createCustomSample(forewardSampleName, outputDataset.id, forewardZip) is None:
        raise RuntimeError(">> [Primer Removal] Failed to upload trimmed foreward reads")

    if pairedEnd:
        reverseZip = folder_manager.temp / "reverseReads.zip"
        with ZipFile(reverseZip, 'w', ZIP_DEFLATED) as archive:
            for filePath in reverseReadsFolder.iterdir():
                archive.write(filePath,filePath.name)

        if CustomSample.createCustomSample("R2_reverse_reads", outputDataset.id, reverseZip) is None:
            raise RuntimeError(">> [Primer Removal] Failed to upload trimmed reverse reads")


if __name__ == "__main__":
    initializeProject(main)
