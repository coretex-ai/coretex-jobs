from pathlib import Path
from typing import Optional

import logging
import os
import shutil
import gc

from ultralytics import YOLO
from coretex import TaskRun, TaskRunStatus, ImageDataset, folder_manager, Model

import torch

from .callback import onTrainStart, onEpochEnd
from .validate import validate
from .dataset import createYamlFile


def saveModelDescriptor(taskRun:TaskRun[ImageDataset], model: Model, path: Path) -> None:
    labels = [
        {
            "label": clazz.label,
            "color": clazz.color
        }
        for clazz in taskRun.dataset.classes
    ]

    model.saveModelDescriptor(path, {
        "taskRunId": taskRun.id,
        "modelName": model.name,
        "spaceName": taskRun.projectName,
        "epochs": taskRun.parameters["epochs"],
        "batchSize": taskRun.parameters["batchSize"],
        "labels": labels,
        "description": taskRun.description,
        "imageSize": taskRun.parameters["imageSize"],

        "input_description": "RGB image",
        "input_shape": taskRun.parameters["imageSize"],

        "outputDescription": "Segmentation mask",
        "outputShape": taskRun.parameters["imageSize"]
    })


def copyToValFolder(sourceFolder: Path, destinationFolder: Path) -> None:
    for item in os.listdir(sourceFolder):
        sourcePath = os.path.join(sourceFolder, item)
        destinationPath = os.path.join(destinationFolder, item)
        shutil.copy2(sourcePath, destinationPath)


def getPatience(taskRun: TaskRun) -> int:
    epochs = taskRun.parameters["epochs"]
    earlyStopping = taskRun.parameters["earlyStopping"]

    if earlyStopping:
        # 10% of epochs or 10, whichever is higher
        return max(int(epochs * 0.1), 10)

    # To disable early stopping we have to set patience to a very high value
    # https://github.com/ultralytics/ultralytics/issues/7609
    return 2 ** 64


def justTrain(taskRun: TaskRun[ImageDataset], yamlFilePath: Path, yoloModelPath: Path) -> Path:
    ctxModel: Optional[Model] = taskRun.parameters.get("model")
    if ctxModel is None:
        # Start training from specified YoloV8 weights
        weights = taskRun.parameters.get("weights", "yolov8n-seg.pt")
        logging.info(f">> [Image Segmentation] Using \"{weights}\" for training the model")

        model = YOLO(taskRun.parameters.get("weights", "yolov8n-seg.pt"))
    else:
        logging.info(f">> [Image Segmentation] Using \"{ctxModel.name}\" for training the model")

        # Start training from specified model checkpoint
        ctxModel.download()
        model = YOLO(ctxModel.path / "best.pt")

    model.add_callback("on_train_start", onTrainStart)
    model.add_callback("on_train_epoch_end", onEpochEnd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f">> [Image Segmentation] The {str(device).upper()} will be used for training")

    logging.info(">> [Image Segmentation] Training the model")
    model.train(
        project = yoloModelPath,
        data = yamlFilePath,
        epochs = taskRun.parameters["epochs"],
        batch = taskRun.parameters["batchSize"],
        imgsz = taskRun.parameters["imageSize"],
        patience = getPatience(taskRun),
        plots = False,
        resume = False
    )
    del model
    gc.collect()

    return yoloModelPath / "train" / "weights" / "best.pt"


def train(taskRun: TaskRun[ImageDataset], datasetPath: Path, trainDatasetPath: Path, validDatasetPath: Path) -> None:
    taskRun.updateStatus(TaskRunStatus.inProgress, "Training")

    yamlFilePath = datasetPath / "config.yaml"
    createYamlFile(datasetPath, trainDatasetPath, validDatasetPath, taskRun.dataset.classes, yamlFilePath)

    yoloModelPath = folder_manager.createTempFolder("yolo_model")
    modelPath = justTrain(taskRun, yamlFilePath, yoloModelPath)

    accuracy = validate(taskRun, modelPath, taskRun.parameters["imageSize"])

    model = YOLO(modelPath)
    model.export(format = "tflite")
    model.export(format = "tfjs")
    model.export(format = "coreml")

    modelName = taskRun.generateEntityName()
    ctxModel = Model.createModel(modelName, taskRun.projectId, accuracy / 100, {})
    saveModelDescriptor(taskRun, ctxModel, modelPath.parent)
    ctxModel.upload(modelPath.parent)
    logging.info(">> [Image Segmentation] The trained model has been uploaded")

    taskRun.submitOutput("outputModel", ctxModel)
