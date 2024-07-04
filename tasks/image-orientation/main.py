import logging

from coretex import currentTaskRun, Model, folder_manager, Metric, MetricType, ImageDataset, TaskRun
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import OrientedDataset, getTransform, prepareDataset, splitDataset
from src.model import OrientationClassifier
from src.train import runTraining
from src.validation import runValidation
from src.utils import getMeanAndStd


def createMetrics(taskRun: TaskRun) -> None:
    taskRun.createMetrics([
        Metric.create("training_loss", "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("validation_loss", "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("training_accuracy", "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("validation_accuracy", "epoch", MetricType.int, "value", MetricType.float)
    ])


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()
    createMetrics(taskRun)
    taskRun.dataset.download()

    logging.info(">> [Orientation] Preparing dataset")
    imagesDir, sampleIds = prepareDataset(taskRun.dataset)

    imageSize = taskRun.parameters["imageSize"]
    epochs = taskRun.parameters["epochs"]
    batchSize = taskRun.parameters["batchSize"]
    labelColumn = taskRun.parameters["labelColumn"]

    mean, std = getMeanAndStd(imagesDir)
    transform = getTransform((imageSize, imageSize), (mean, std))

    dataset = OrientedDataset(imagesDir, sampleIds, labelColumn, transform)
    trainDataset, validDataset = splitDataset(dataset, taskRun.parameters["validSplit"])

    trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle = True, num_workers = 0)
    validLoader = DataLoader(validDataset, batch_size = batchSize, shuffle = True, num_workers = 0)

    logging.info(">> [Orientation] Building model")
    model = OrientationClassifier()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = taskRun.parameters["lr"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.is_available():
        logging.info(">> [Orientation] Using GPU for training")
    else:
        logging.info(">> [Orientation] Using CPU for training")

    modelPath = folder_manager.createTempFolder("model")

    logging.info(">> [Orientation] Starting training")
    runTraining(
        model,
        trainLoader,
        validLoader,
        optimizer,
        criterion,
        device,
        epochs,
        taskRun,
        modelPath,
        imageSize
    )
    logging.info(">> [Orientation] Finished Training")

    logging.info(">> [Orientation] Running validation")
    accuracy = runValidation(modelPath / "best.pt", trainLoader, validLoader, device, taskRun)

    logging.info(">> [Orientation] Uploading model to Coretex")
    ctxModel = Model.createModel(f"{taskRun.id}-orientation-model", taskRun.id, accuracy, {})
    Model.saveModelDescriptor(modelPath, {
        "taskRunId": taskRun.id,
        "modelName": taskRun.name,
        "spaceName": taskRun.projectName,
        "projectName": taskRun.taskName,
        "epochs": epochs,
        "batchSize": batchSize,
        "imageSize": imageSize,
        "mean": mean,
        "std": std
    })
    ctxModel.upload(modelPath)

    taskRun.submitOutput("orientationModel", ctxModel)


if __name__ == "__main__":
    main()
