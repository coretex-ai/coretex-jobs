import csv
import logging

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import numpy as np

from coretex import Run, RunStatus, folder_manager

from .utils import getKey


def savePredictionFile(
    run: Run,
    predictions: np.ndarray,
    outputMatrix: np.ndarray,
    sampleIdList: list,
    uniqueBodySite: dict[str, int]
) -> None:

    predictionFilePath = folder_manager.temp / "body_site_predictions.csv"

    with open(predictionFilePath, "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_ids", "body-site", "body-site-prediction"])
        for i in range(len(outputMatrix)):
            writer.writerow([
                sampleIdList[i],
                getKey(uniqueBodySite, outputMatrix[i]),
                getKey(uniqueBodySite, predictions[i])
            ])

    run.createArtifact(predictionFilePath, "body_site_predictions.csv")


def validate(
    run: Run,
    inputMatrix: np.ndarray,
    output: np.ndarray,
    uniqueBodySites: dict[str, int],
    sampleIdList: list,
    trainedModelId: int
) -> None:

    trainedModelPath = folder_manager.modelsFolder / str(trainedModelId)

    run.updateStatus(RunStatus.inProgress, "Running validation with pretrained XGBoost model")
    logging.info(">> [MicrobiomeForensics] Starting validation with pretrained XGBClassifier model")

    model = XGBClassifier()
    model.load_model(trainedModelPath / "model.txt")

    predictions = model.predict(inputMatrix)

    accuracy = accuracy_score(output, predictions)
    logging.info(f">> [MicrobiomeForensics] Validation finished, accuracy: {round(accuracy * 100, 2)}%")

    savePredictionFile(
        run,
        predictions,
        output,
        sampleIdList,
        uniqueBodySites
    )
