from pathlib import Path

import os
import json
import glob
import logging

import numpy as np

from coretex import Experiment, CustomDataset, ExperimentStatus
from coretex.folder_management import FolderManager

from .utils import savePlotFig


def loadDataStd(dataset: CustomDataset, experiment: Experiment[CustomDataset]) -> tuple[int, int, dict[str, int], dict[str, int], list[int]]:
    logging.info(">> [MicrobiomeForensics] Downloading dataset...")
    experiment.updateStatus(ExperimentStatus.inProgress, "Downloading dataset...")
    dataset.download()

    experiment.updateStatus(ExperimentStatus.inProgress, "Loading data")

    tempPath = Path(FolderManager.instance().temp)

    taxonDistributionSavePath = tempPath / "taxon_histogram.png"
    classDistributionSavePath = tempPath / "body_site_histogram.png"

    datasetLen = len(experiment.dataset.samples)
    if datasetLen < 10:
        raise RuntimeError(f">> [MicrobiomeForensics] You have insufficient number of samples in your dataset ({datasetLen})")

    level: int = experiment.parameters["taxonomicLevel"]

    uniqueTaxons: dict[str, int] = {}
    uniqueBodySites: dict[str, int] = {}

    taxonDistribution: dict[str, int] = {}
    classDistribution: dict[str, int] = {}

    for sample in dataset.samples:
        sample.unzip()
        samplePath = glob.glob(os.path.join(sample.path, f"*.json"))[0]

        with open(samplePath, "r") as f:
            sample = json.load(f)

        if not sample["body_site"] in uniqueBodySites:
            uniqueBodySites[sample["body_site"]] = len(uniqueBodySites)
            classDistribution[sample["body_site"]] = 1
        else:
            classDistribution[sample["body_site"]] += 1

        for bacteria in sample["97"]:
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
        experiment,
        classDistribution,
        classDistributionSavePath,
        "body_site_histogram.png",
        False,
        "Body Site",
        "Number of samples",
        "Class distribution"
    )

    savePlotFig(
        experiment,
        taxonDistribution,
        taxonDistributionSavePath,
        "taxon_histogram.png",
        True,
        "Taxon",
        "Total count",
        "Representation of dataset sample"
    )

    return level, datasetLen, uniqueTaxons, uniqueBodySites


def prepareForTrainingStd(level: int, datasetLen: int, uniqueTaxons: dict, uniqueBodySites: dict, experiment: Experiment[CustomDataset]) -> tuple[np.ndarray, np.ndarray]:
    inputMatrix = np.zeros((datasetLen, len(uniqueTaxons)))
    outputMatrix = np.zeros((datasetLen, 1))

    logging.info(">> [MicrobiomeForensics] Preparing data for training. Generating 2 matrices, input and output")
    logging.info(f">> [MicrobiomeForensics] Input matrix shape: {inputMatrix.shape}. Output matrix shape: {outputMatrix.shape}")

    sampleIdList = []

    for i, sample in enumerate(experiment.dataset.samples):
        sample.unzip()
        samplePath = glob.glob(os.path.join(sample.path, f"*.json"))[0]

        with open(samplePath, "r") as f:
            sample = json.load(f)

        for bacteria in sample["97"]:
            sampleIdList.append(sample["_id"]["$oid"])

            taxons = bacteria["taxon"].split(";")
            taxon = taxons[level]
            encodedTaxon = uniqueTaxons[taxon]

            c = bacteria["count"]
            inputMatrix[i, encodedTaxon] += c

            outputMatrix[i, 0] = uniqueBodySites[sample["body_site"]]

    return inputMatrix, outputMatrix, sampleIdList
