from pathlib import Path

import logging

from coretex import Experiment, CustomDataset, folder_manager
from coretex.project import initializeProject
from coretex.bioinformatics import sequence_alignment as sa

from src.index import index
from src.sequence_alignment import alignToRefDatabase, sam2bam
from src.separate import separate
from src.utils import loadIndexed, clearDirectory, uploadToCoretex, prepareGroups
from src.filepaths import BWA, SAMTOOLS


def main(experiment: Experiment[CustomDataset]) -> None:
    outDir = Path(folder_manager.createTempFolder("output"))
    samDir = Path(folder_manager.createTempFolder("sam"))
    bamDir = Path(folder_manager.createTempFolder("bam"))

    groupNames: list[str] = experiment.parameters["separationGroups"]
    thresholds: list[int] = experiment.parameters["separationThresholds"]

    sa.chmodX(Path(BWA))
    sa.chmodX(Path(SAMTOOLS))

    if experiment.parameters["useBacteriaLib"]:
        raise NotImplementedError(">> [Region Separation] useBacteriaLib has not been implemented yet, use dataset 7153 with referenceDatasetIndexed as True insted")

    logging.info(">> [Region Separation] Downloading dataset")
    experiment.dataset.download()
    inputFiles = sa.loadFa(experiment.dataset)

    if not experiment.parameters["referenceDatasetIndexed"]:
        logging.info(">> [Region Separation] Index reference genome")
        referenceDirs = index(experiment.parameters["referenceDataset"])

    else:
        logging.info(">> [Region Separation] Loading indexed reference dataset")
        referenceDirs = loadIndexed(experiment.parameters["referenceDataset"])

    groups, thresholds = prepareGroups(groupNames, thresholds, outDir)

    for inputFile in inputFiles:
        logging.info(f">> [Region Separation] Starting alignment for {inputFile.name}")
        alignToRefDatabase(inputFile, referenceDirs, samDir)

        logging.info(f">> [Region Separation] Starting conversion to binary format for {inputFile.name}")
        sam2bam(samDir, bamDir)

        logging.info(f">> [Region Separation] Starting read separation into groups for {inputFile.name}")
        separate(bamDir, inputFile, groups, thresholds, experiment.parameters["newReadIndicator"])

        clearDirectory(samDir)
        clearDirectory(bamDir)

    uploadToCoretex(experiment, groups)


if __name__ == "__main__":
    initializeProject(main)
