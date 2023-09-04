from typing import Optional
from pathlib import Path

import csv
import json
import logging

from scipy.sparse import csr_matrix

import numpy as np
import matplotlib.pyplot as plt

from coretex import CustomDataset, Run, Model, folder_manager

from .objects import Sample


def jsonPretty(data, savePath) -> None:
    with open(savePath, "w") as write_file:
        json.dump(data, write_file, indent=4)


def saveModel(accuracy: float, uniqueBodySites: dict[str, int], lenOfData: int, numOfUniqueTaxons: int, run: Run[CustomDataset]) -> None:
    modelPath = folder_manager.temp / "modelFolder"

    labels = list(uniqueBodySites.keys())

    model = Model.createModel(run.name, run.id, accuracy, {})
    model.saveModelDescriptor(modelPath, {
        "project_task": run.spaceTask,
        "labels": labels,
        "modelName": model.name,
        "description": run.description,

        "input_description": """
            Input shape is [len(dataOfSamples), len(listOfUniqueTaxons)]

            - len(dataOfSamples) is actually number of samples in dataset
            - len(listOfUniqueTaxons) represents number of unique taxons for selected level in dataset
        """,
        "input_shape": [lenOfData, numOfUniqueTaxons],

        "output_description": """
            Output shape - [len(dataOfSamples), predictedClass]

            - len(dataOfSamples) is actually number of samples in dataset
            - 1 represents that output 2d array (table) is going to have only 1 column (1 prediction for each sample in dataset)
        """,
        "output_shape": [lenOfData, 1]
    })

    model.upload(modelPath)


def getKey(dictionary: dict[str, int], val: int) -> Optional[str] :
    for key, value in dictionary.items():
        if val == value:
            return key

    return None


def getBodySite(lineId: str, dataDict: dict[str, str]) -> Optional[tuple[str, str]]:
    value = dataDict.get(lineId)
    if value is None:
        return None

    valueSplit = value.split(",")
    return valueSplit[0], valueSplit[1]


def saveFeatureTable(featureTablePath: str, tableInput: np.ndarray, run: Run[CustomDataset]) -> None:
    np.savetxt(featureTablePath, tableInput, delimiter=",", fmt = "%i")
    run.createArtifact(featureTablePath, "feature_table.csv")


def savePlotFig(
    run: Run[CustomDataset],
    distributionDict: dict,
    savePath: str,
    fileName: str,
    xLabelRotation: bool,
    xLabel: str,
    yLabel: str,
    plotTitle: str
) -> None:

    distributionDict = dict(sorted(distributionDict.items())) # Sort by taxon id
    distributionDictKeys = list(distributionDict.keys())
    distributionDictValues = list(distributionDict.values())

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize = (18, 9))

    plt.bar(range(len(distributionDict)), distributionDictValues, tick_label = distributionDictKeys,  color = "blue", width = 0.5)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(plotTitle)

    if xLabelRotation:
        plt.xticks(rotation = 45, ha = "right")

    plt.savefig(savePath, bbox_inches = "tight")
    run.createArtifact(savePath, fileName)


def savePredictionFile(
    run: Run[CustomDataset],
    savePath: str,
    xTrain: csr_matrix,
    xTest: csr_matrix,
    sampleIdList: list,
    uniqueBodySite: dict,
    yTrain: list,
    yTest: list,
    yPred: list,
    zPred: list
) -> None:

    with open(folder_manager.temp / "body_site_predictions.csv", "a+") as f:
        z = 0
        i = 0
        writer = csv.writer(f)
        writer.writerow(["sample_ids", "body-site", "body-site-prediction"])
        while i < xTrain.shape[0]:
            # writer.writerow([sampleIdList[i],  getKey(uniqueBodySite, yTrain[i]), getKey(uniqueBodySite, zPred[i])])
            i += 1
            z = i
        else:
            for i in range(xTest.shape[0]):
                writer.writerow([
                    sampleIdList[z],
                    getKey(uniqueBodySite, yTest[i]),
                    getKey(uniqueBodySite, yPred[i])
                ])
                z += 1

    run.createArtifact(savePath, "body_site_predictions.csv")


def validateDataset(dataset: CustomDataset) -> None:
    if len(dataset.samples) != 1:
        raise ValueError(">> [MicrobiomeForensics] Dataset must have only one sample")

    zippedSample = dataset.samples[0]
    zippedSample.unzip()

    if not any([sample.suffix == ".mapped" for sample in Path(zippedSample.path).iterdir()]) or not any([sample.suffix == ".info" for sample in Path(zippedSample.path).iterdir()]):
        raise ValueError(">> Invalid Dataset. Dataset sample must contain a \".mapped\" file and \"samples.env.info\"")


def calculateDistributions(sampleData: list[Sample]) -> tuple[dict[str, int], dict[str, int]]:
    classDistribution: dict[str, int] = {}
    taxonDistribution: dict[str, int] = {}

    for element in sampleData:
        bodySite = element.bodySite
        if bodySite in classDistribution:
            classDistribution[bodySite] += 1
        else:
            classDistribution[bodySite] = 1

        for taxonData in element.taxons:
            taxonId = taxonData.taxonId
            taxonCount = taxonData.count

            if not taxonData.taxonId in taxonDistribution:
                taxonDistribution[taxonId] = taxonCount
            else:
                taxonDistribution[taxonId] += taxonCount

    return classDistribution, taxonDistribution


def plots(sampleData: list[Sample], run: Run[CustomDataset]) -> None:

    """
        Creates taxon_histogram.png and body_site_histogram.png in run artifacts.

        Parameters
        ----------
        sampleData: list[Sample]
            The list of all samples (the dataset)
    """

    taxonDistributionSavePath = folder_manager.temp / "taxon_histogram.png"
    classDistributionSavePath = folder_manager.temp / "body_site_histogram.png"

    classDistribution, taxonDistribution = calculateDistributions(sampleData)

    taxonDistributionValues = list(taxonDistribution.values())
    taxonDistribution = {k:v for k, v in taxonDistribution.items() if v > (max(taxonDistributionValues)/10)}

    if len(sampleData) < 10:
        raise RuntimeError(f">> [MicrobiomeForensics] You have insufficient number of samples in your dataset ({len(sampleData)})")

    savePlotFig(
        run,
        classDistribution,
        classDistributionSavePath,
        "body_site_histogram.png",
        True,
        "Body Site",
        "Number of samples",
        "Class distribution"
    )

    savePlotFig(
        run,
        taxonDistribution,
        taxonDistributionSavePath,
        "taxon_histogram.png",
        True,
        "Taxon",
        "Total count",
        "Representation of dataset sample"
    )

    logging.info(f">> [MicrobiomeForensics] Loading data and matching finished. Successfully matched {len(sampleData)} samples")
