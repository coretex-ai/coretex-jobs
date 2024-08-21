import logging
import csv

from coretex import TaskRun, ImageDataset, TaskRunStatus, folder_manager
from keras import Model as KerasModel

import numpy as np

from .detect import predict


def createDatasetResults(taskRun: TaskRun[ImageDataset], sampleResults: list[dict[str, str]]) -> None:
    if len(sampleResults) <= 0:
        raise RuntimeError("There are no processed prediction results")

    datasetAcc = [float(result['accuracy']) for result in sampleResults]

    datasetResultPath = folder_manager.temp / "dataset_results.csv"
    with datasetResultPath.open("w") as file:
        writer = csv.DictWriter(file, fieldnames = ["Class", "Accuracy", "Accuracy stdev"])
        writer.writeheader()

        for label in taskRun.dataset.classes.labels:
            classAcc = [float(result[label]) for result in sampleResults]
            writer.writerow({
                "Class": label,
                "Accuracy": f"{np.mean(classAcc):.2f}",
                "Accuracy stdev": f"{np.std(classAcc):.2f}"
            })

        writer.writerow({
        "Class": "dataset",
        "Accuracy": f"{np.mean(datasetAcc):.2f}",
        "Accuracy stdev": f"{np.std(datasetAcc):.2f}"
    })

    if taskRun.createArtifact(datasetResultPath, datasetResultPath.name) is None:
        logging.warning(">> [Image Segmentation] Failed to upload csv file with dataset results as artifact")
    else:
        logging.info(">> [Image Segmentation] The csv file with dataset results has been uploaded as artifact")


def createSampleResults(taskRun: TaskRun[ImageDataset], sampleResults: list[dict[str, str]]) -> None:
    if len(sampleResults) <= 0:
        raise RuntimeError("There are no processed prediction results")

    csvSamplesPath = folder_manager.temp / "sample_results.csv"
    with csvSamplesPath.open("w") as file:
        writer = csv.DictWriter(file, fieldnames = ["sample id", "sample name", "accuracy"])
        writer.writeheader()

        for result in sampleResults:
            writer.writerow({index: result[index] for index in ["sample id", "sample name", "accuracy"]})

    if taskRun.createArtifact(csvSamplesPath, csvSamplesPath.name) is None:
        logging.warning(">> [Image Segmentation] Failed to upload csv file with samples results as artifact")
    else:
        logging.info(">> [Image Segmentation] The csv file with samples results has been uploaded as artifact")


def batchPredict(taskRun: TaskRun[ImageDataset], model: KerasModel, batchSize: int) -> list[dict[str, str]]:
    batchResult: list[dict[str, str]] = []
    for index in range(0, taskRun.dataset.count, batchSize):
        batchResult.extend(predict(taskRun, model, taskRun.dataset.samples[index:index + batchSize]))

    return batchResult


def validate(taskRun: TaskRun[ImageDataset], model: KerasModel) -> float:
    taskRun.updateStatus(TaskRunStatus.inProgress, "Validating")

    predictionResults: list[dict[str, str]] = []
    predictionResults.extend(batchPredict(taskRun, model, taskRun.parameters["batchSize"]))

    taskRun.updateStatus(TaskRunStatus.inProgress, "Creating csv fils with results")
    createSampleResults(taskRun, predictionResults)
    createDatasetResults(taskRun, predictionResults)
    datasetAcc = [float(result['accuracy']) for result in predictionResults]

    return float(np.mean(datasetAcc))
