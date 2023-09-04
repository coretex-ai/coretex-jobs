from typing import Optional
from pathlib import Path

import csv
import json
import logging

from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt

from coretex import CustomDataset, Run, Model, folder_manager


def jsonPretty(data, savePath) -> None:
    with open(savePath, "w") as write_file:
        json.dump(data, write_file, indent=4)


def saveModel(run: Run[CustomDataset], accuracy: float, uniqueBodySites: dict[str, int], lenOfData: int, numOfUniqueTaxons: int) -> None:
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


def saveFeatureTable(run: Run[CustomDataset], featureTablePath: str, tableInput: np.ndarray) -> None:
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

    distributionDict = dict(sorted(distributionDict.items()))  # Sort by taxon id
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
    trainCount: int,
    testCount: int,
    sampleIds: list,
    uniqueBodySite: dict,
    yTrain: list,
    yTest: list,
    yPred: list,
    zPred: list
) -> None:

    with folder_manager.temp.joinpath("body_site_predictions.csv").open("a+") as f:
        z = 0
        i = 0
        writer = csv.writer(f)
        writer.writerow(["sample_ids", "body-site", "body-site-prediction"])

        while i < trainCount:
            # writer.writerow([sampleIds[i],  getKey(uniqueBodySite, yTrain[i]), getKey(uniqueBodySite, zPred[i])])
            i += 1
            z = i
        else:
            for i in range(testCount):
                writer.writerow([
                    sampleIds[z],
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


def plots(run: Run[CustomDataset], classDistribution: dict[str, int], taxonDistribution: dict[str, int], datasetLen: int) -> None:

    """
        Creates taxon_histogram.png and body_site_histogram.png in run artifacts.

        Parameters
        ----------
        classDistribution : dict[str, int]
            A dictonary mapping each class to the number of times it occurs in the dataset
        taxonDistribution : dict[str, int]
            A dictonary mapping each taxon id to its total count in the dataset
        datasetLen : int
            The number of samples in the dataset
    """

    taxonDistributionSavePath = folder_manager.temp / "taxon_histogram.png"
    classDistributionSavePath = folder_manager.temp / "body_site_histogram.png"

    taxonDistributionValues = list(taxonDistribution.values())
    taxonDistribution = {k:v for k, v in taxonDistribution.items() if v > (max(taxonDistributionValues)/10)}

    if datasetLen < 10:
        raise RuntimeError(f">> [MicrobiomeForensics] You have insufficient number of samples in your dataset ({datasetLen})")

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

    logging.info(f">> [MicrobiomeForensics] Loading data and matching finished. Successfully matched {datasetLen} samples")


def oneHotEncoding(vector: ArrayLike, num_classes: Optional[int] = None) -> np.ndarray:

    """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Parameters
        ----------
        vector : ArrayLike
            A vector of integers
        num_classes : int
            Optionally declare the number of classes (can not exceed the maximum value of the vector)

        Returns
        -------
        result : np.ndarray
            The one hot encoded vecotr or matrix

        Example
        -------
        >>> v = np.array((1, 0, 4))
        >>> one_hot_v = oneHotEncoding(v)
        >>> print one_hot_v
        [[0 1 0 0 0]
        [1 0 0 0 0]
        [0 0 0 0 1]]
    """

    vecLen = 1 if isinstance(vector, int) else len(vector)

    result = np.zeros(shape = (vecLen, num_classes))
    result[np.arange(vecLen), vector] = 1
    return result.astype(int)


def convertFromOneHot(matrix: np.ndarray) -> np.ndarray:
    numOfRows = len(matrix) if isinstance(matrix, list) else matrix.shape[0]
    if not numOfRows > 0:
        raise RuntimeError(f">> [MicrobiomeForensics] Encountered array with {numOfRows} rows when decoding one hot vector")

    result = np.empty(shape = (numOfRows, ), dtype = np.int32)
    for i, sample in enumerate(matrix):
        result[i] = sample.argmax()

    return result.astype(np.int32)
