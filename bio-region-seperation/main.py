from pathlib import Path

import logging

from coretex import Experiment, CustomDataset
from coretex.folder_management import FolderManager
from coretex.project import initializeProject
from coretex.bioinformatics.sequence_alignment import loadFa, chmodX

from src.index import index
from src.sequence_alignment import alignToRefDatabase, sam2bam
from src.separate import separate
from src.utils import loadIndexed, clearDirectory, uploadToCoretex, prepareGroups
from src.filepaths import BWA, SAMTOOLS


def main(experiment: Experiment[CustomDataset]) -> None:
    outDir = Path(FolderManager.instance().createTempFolder("output"))
    samDir = Path(FolderManager.instance().createTempFolder("sam"))
    bamDir = Path(FolderManager.instance().createTempFolder("bam"))

    groupNames: list[str] = experiment.parameters["separationGroups"]
    thresholds: list[int] = experiment.parameters["separationThresholds"]

    chmodX(Path(BWA))
    chmodX(Path(SAMTOOLS))

    if experiment.parameters["useBacteriaLib"]:
        raise NotImplementedError(">> [Sequence Alignment] useBacteriaLib has not been implemented yet, use dataset 7153 with referenceDatasetIndexed as True insted")

    logging.info(">> [Sequence Alignment] Downloading dataset")
    experiment.dataset.download()
    inputFiles = loadFa(experiment.dataset)

    if not experiment.parameters["referenceDatasetIndexed"]:
        logging.info(">> [Sequence Alignment] Index reference genome")
        referenceDirs = index(experiment.parameters["referenceDataset"])

    else:
        logging.info(">> [Sequence Alignment] Loading indexed reference dataset")
        referenceDirs = loadIndexed(experiment.parameters["referenceDataset"])

    groups, thresholds = prepareGroups(groupNames, thresholds, outDir)

    for inputFile in inputFiles:
        logging.info(f">> [Sequence Alignment] Starting alignment for {inputFile.name}")
        alignToRefDatabase(inputFile, referenceDirs, samDir)

        logging.info(f">> [Sequence Alignment] Starting conversion to binary format for {inputFile.name}")
        sam2bam(samDir, bamDir)

        logging.info(f">> [Sequence Alignment] Starting read separation into groups for {inputFile.name}")
        separate(bamDir, inputFile, groups, thresholds, experiment.parameters["newReadIndicator"])

        clearDirectory(samDir)
        clearDirectory(bamDir)

    uploadToCoretex(experiment, groups)


if __name__ == "__main__":
    initializeProject(main)
