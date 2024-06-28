import os
import json
import glob
import logging

import numpy as np

from coretex import TaskRun, CustomDataset, TaskRunStatus, folder_manager

from .utils import savePlotFig


def loadDataStd(dataset: CustomDataset, taskRun: TaskRun[CustomDataset]) -> tuple[int, int, dict[str, int], dict[str, int]]:
    logging.info(">> [MicrobiomeForensics] Downloading dataset...")
    taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset...")
    dataset.download()

    taskRun.updateStatus(TaskRunStatus.inProgress, "Loading data")

    taxonDistributionSavePath = folder_manager.temp / "taxon_histogram.png"
    classDistributionSavePath = folder_manager.temp / "body_site_histogram.png"

    datasetLen = len(taskRun.dataset.samples)
    if datasetLen < 10:
        raise RuntimeError(f">> [MicrobiomeForensics] You have insufficient number of samples in your dataset ({datasetLen})")

    level: int = taskRun.parameters["taxonomicLevel"]

    uniqueTaxons: dict[str, int] = {}
    uniqueBodySites: dict[str, int] = {}

    taxonDistribution: dict[str, int] = {}
    classDistribution: dict[str, int] = {}

    for sample in dataset.samples:
        sample.unzip()
        samplePath = glob.glob(os.path.join(sample.path, f"*.json"))[0]

        with open(samplePath, "r") as f:
            sampleDict = json.load(f)

        if not sampleDict["body_site"] in uniqueBodySites:
            uniqueBodySites[sampleDict["body_site"]] = len(uniqueBodySites)
            classDistribution[sampleDict["body_site"]] = 1
        else:
            classDistribution[sampleDict["body_site"]] += 1

        for bacteria in sampleDict["97"]:
            taxons = bacteria["taxon"].split(";")
            taxon = taxons[level]

            if not taxon in uniqueTaxons:
                uniqueTaxons[taxon] = len(uniqueTaxons)
                taxonDistribution[taxon] = bacteria["count"]
            else:
                taxonDistribution[taxon] += bacteria["count"]

    taxonDistributionValues = list(taxonDistribution.values())
    taxonDistribution = {k:v for k, v in taxonDistribution.items() if v > (max(taxonDistributionValues)/10)}

    savePlotFig(
        taskRun,
        classDistribution,
        classDistributionSavePath,
        "body_site_histogram.png",
        True,
        "Body Site",
        "Number of samples",
        "Class distribution"
    )

    savePlotFig(
        taskRun,
        taxonDistribution,
        taxonDistributionSavePath,
        "taxon_histogram.png",
        True,
        "Taxon",
        "Total count",
        "Representation of dataset sample"
    )

    return level, datasetLen, uniqueTaxons, uniqueBodySites


def prepareForTrainingStd(level: int, datasetLen: int, uniqueTaxons: dict, uniqueBodySites: dict, taskRun: TaskRun[CustomDataset]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    inputMatrix = np.zeros((datasetLen, len(uniqueTaxons)))
    outputMatrix = np.zeros((datasetLen, 1))

    logging.info(">> [MicrobiomeForensics] Preparing data for training. Generating 2 matrices, input and output")
    logging.info(f">> [MicrobiomeForensics] Input matrix shape: {inputMatrix.shape}. Output matrix shape: {outputMatrix.shape}")

    sampleIdList = []

    for i, sample in enumerate(taskRun.dataset.samples):
        sample.unzip()
        samplePath = glob.glob(os.path.join(sample.path, f"*.json"))[0]

        with open(samplePath, "r") as f:
            sampleDict = json.load(f)

        for bacteria in sampleDict["97"]:
            sampleIdList.append(sampleDict["_id"]["$oid"])

            taxons = bacteria["taxon"].split(";")
            taxon = taxons[level]
            encodedTaxon = uniqueTaxons[taxon]

            c = bacteria["count"]
            inputMatrix[i, encodedTaxon] += c

            outputMatrix[i, 0] = uniqueBodySites[sampleDict["body_site"]]

    return inputMatrix, outputMatrix, sampleIdList
