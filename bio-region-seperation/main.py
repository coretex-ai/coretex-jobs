from pathlib import Path

import logging

from coretex import TaskRun, CustomDataset, folder_manager, currentTaskRun
from coretex.bioinformatics import sequence_alignment as sa

from src.index import index
from src.sequence_alignment import alignToRefDatabase, sam2bam
from src.separate import separate
from src.utils import loadIndexed, clearDirectory, uploadToCoretex, prepareGroups
from src.filepaths import BWA, SAMTOOLS


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    outDir = Path(folder_manager.createTempFolder("output"))
    samDir = Path(folder_manager.createTempFolder("sam"))
    bamDir = Path(folder_manager.createTempFolder("bam"))

    groupNames: list[str] = taskRun.parameters["separationGroups"]
    thresholds: list[int] = taskRun.parameters["separationThresholds"]

    sa.chmodX(Path(BWA))
    sa.chmodX(Path(SAMTOOLS))

    logging.info(">> [Region Separation] Downloading dataset")
    taskRun.dataset.download()
    inputFiles = sa.loadFa(taskRun.dataset)

    if not taskRun.parameters["referenceDatasetIndexed"]:
        logging.info(">> [Region Separation] Index reference genome")
        referenceDirs = index(taskRun.parameters["referenceDataset"])

    else:
        logging.info(">> [Region Separation] Loading indexed reference dataset")
        referenceDirs = loadIndexed(taskRun.parameters["referenceDataset"])

    groups, thresholds = prepareGroups(groupNames, thresholds, outDir)

    for inputFile in inputFiles:
        logging.info(f">> [Region Separation] Starting alignment for {inputFile.name}")
        alignToRefDatabase(inputFile, referenceDirs, samDir)

        logging.info(f">> [Region Separation] Starting conversion to binary format for {inputFile.name}")
        sam2bam(samDir, bamDir)

        logging.info(f">> [Region Separation] Starting read separation into groups for {inputFile.name}")
        separate(bamDir, inputFile, groups, thresholds)

        clearDirectory(samDir)
        clearDirectory(bamDir)

    uploadToCoretex(taskRun, groups)


if __name__ == "__main__":
    main()
