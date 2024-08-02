from pathlib import Path

import logging

from ultralytics import YOLO
from coretex import TaskRun, ImageDataset, folder_manager, Model

from .dataset import isValidationSplitValid, prepareDataset, createYamlFile
from .callback import onTrainStart, onEpochEnd
from .validate import validate


def getPatience(taskRun: TaskRun) -> int:
    epochs = taskRun.parameters["epochs"]
    earlyStopping = taskRun.parameters["earlyStopping"]

    if earlyStopping:
        # 10% of epochs or 10, whichever is higher
        return max(int(epochs * 0.1), 10)

    # To disable early stopping we have to set patience to a very high value
    # https://github.com/ultralytics/ultralytics/issues/7609
    return 2 ** 64


def train(taskRun: TaskRun[ImageDataset]) -> None:
    if not isValidationSplitValid(taskRun.parameters["validationSplit"], taskRun.dataset.count):
        raise ValueError(f">> [ObjectDetection] validationSplit parameter is invalid")

    datasetPath = folder_manager.createTempFolder("dataset")
    trainDatasetPath, validDatasetPath = prepareDataset(taskRun.dataset, datasetPath, taskRun.parameters["validationSplit"])

    yamlFilePath = datasetPath / "config.yaml"
    createYamlFile(datasetPath, trainDatasetPath, validDatasetPath, taskRun.dataset.classes, yamlFilePath)

    model = YOLO("yolov8n-seg.pt")

    model.add_callback("on_train_start", onTrainStart)
    model.add_callback("on_train_epoch_end", onEpochEnd)

    logging.info(">> [Image Segmentation] Training the model")
    model.train(
        data = yamlFilePath,
        epochs = taskRun.parameters["epochs"],
        batch = taskRun.parameters["batchSize"],
        imgsz = taskRun.parameters["imageSize"],
        patience = getPatience(taskRun)
    )

    logging.info(">> [Image Segmentation] Validating the model")
    model.val()

    accuracy = validate(taskRun)

    modelName = taskRun.generateEntityName()
    onnxModelPath = model.export(format = "onnx")
    ctxModel = Model.createModel(modelName, taskRun.projectId, accuracy / 100, {})
    ctxModel.upload(Path(onnxModelPath).parent)
    logging.info(">> [Image Segmentation] The trained model has been uploaded")

    taskRun.submitOutput("outputModel", ctxModel)
