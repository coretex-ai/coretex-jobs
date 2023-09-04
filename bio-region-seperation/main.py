from pathlib import Path

import logging

from coretex import Run, CustomDataset, folder_manager
from coretex.job import initializeJob
from coretex.bioinformatics import sequence_alignment as sa

from src.index import index
from src.sequence_alignment import alignToRefDatabase, sam2bam
from src.separate import separate
from src.utils import loadIndexed, clearDirectory, uploadToCoretex, prepareGroups
from src.filepaths import BWA, SAMTOOLS


def main(run: Run[CustomDataset]) -> None:
    outDir = Path(folder_manager.createTempFolder("output"))
    samDir = Path(folder_manager.createTempFolder("sam"))
    bamDir = Path(folder_manager.createTempFolder("bam"))

    groupNames: list[str] = run.parameters["separationGroups"]
    thresholds: list[int] = run.parameters["separationThresholds"]

    sa.chmodX(Path(BWA))
    sa.chmodX(Path(SAMTOOLS))

    if run.parameters["useBacteriaLib"]:
        raise NotImplementedError(">> [Region Separation] useBacteriaLib has not been implemented yet, use dataset 7153 with referenceDatasetIndexed as True insted")

    logging.info(">> [Region Separation] Downloading dataset")
    run.dataset.download()
    inputFiles = sa.loadFa(run.dataset)

    if not run.parameters["referenceDatasetIndexed"]:
        logging.info(">> [Region Separation] Index reference genome")
        referenceDirs = index(run.parameters["referenceDataset"])

    else:
        logging.info(">> [Region Separation] Loading indexed reference dataset")
        referenceDirs = loadIndexed(run.parameters["referenceDataset"])

    groups, thresholds = prepareGroups(groupNames, thresholds, outDir)

    for inputFile in inputFiles:
        logging.info(f">> [Region Separation] Starting alignment for {inputFile.name}")
        alignToRefDatabase(inputFile, referenceDirs, samDir)

        logging.info(f">> [Region Separation] Starting conversion to binary format for {inputFile.name}")
        sam2bam(samDir, bamDir)

        logging.info(f">> [Region Separation] Starting read separation into groups for {inputFile.name}")
        separate(bamDir, inputFile, groups, thresholds, run.parameters["newReadIndicator"])

        clearDirectory(samDir)
        clearDirectory(bamDir)

    uploadToCoretex(run, groups)


if __name__ == "__main__":
    initializeJob(main)
