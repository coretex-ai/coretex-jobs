from pathlib import Path

import csv
import logging

from sklearn.metrics import accuracy_score

import numpy as np

from coretex import Run, RunStatus, CustomDataset, folder_manager

from .utils import getKey
from .model import Model
from .dataset import loadDataset, createBatches


def savePredictionFile(run: Run[CustomDataset], predictions: np.ndarray, trueLabels: np.ndarray, sampleIds: list, uniqueBodySite: dict[str, int]) -> None:
    predictionFilePath = folder_manager.temp / "body_site_predictions.csv"

    with predictionFilePath.open("a+") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_ids", "body-site", "body-site-prediction"])
        for i in range(len(trueLabels)):
            writer.writerow([sampleIds[i],  getKey(uniqueBodySite, trueLabels[i]), getKey(uniqueBodySite, predictions[i])])

    run.createArtifact(predictionFilePath, "body_site_predictions.csv")


def validate(run: Run[CustomDataset], datasetPath: Path, uniqueBodySites: dict[str, int], uniqueTaxons: dict[str, int], trainedModelId: int) -> None:
    trainedModelPath = folder_manager.modelsFolder / str(trainedModelId)

    run.updateStatus(RunStatus.inProgress, "Running validation with pretrained LSPIN model")

    batchSize = run.parameters["batchSize"]

    sampleCount = len(list(datasetPath.iterdir()))

    dataset = loadDataset(datasetPath, uniqueBodySites, uniqueTaxons)
    _, _, testData, testBatches = createBatches(dataset, sampleCount, 1, 1, batchSize)

    logging.info(f">> [MicrobiomeForensics] Starting validation with pretrained LSPIN model on {sampleCount} samples with {len(uniqueTaxons)} features")

    model = Model.load(trainedModelPath / "model")

    yPred, yTrue = model.test(testData, testBatches)

    accuracy = accuracy_score(yTrue, yPred)

    sampleIds: list[str] = []
    for path in datasetPath.iterdir():
        sampleIds.append(path.name)

    logging.info(f">> [MicrobiomeForensics] Validation finished, accuracy: {round(accuracy * 100, 2)}%")

    savePredictionFile(
        run,
        yPred,
        yTrue,
        sampleIds,
        uniqueBodySites
    )
