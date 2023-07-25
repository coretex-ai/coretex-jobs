from pathlib import Path

import os
import shutil
import logging

import numpy as np

from coretex import ComputerVisionDataset, Experiment, Model, augmentDataset, Metric, MetricType, folder_manager
from coretex.project import initializeProject

import src.train as train
import src.detect as detect
import src.export as export


def prepareModelForUpload(source: str, destination: str):
    os.rename(source, destination)
    shutil.move(destination, folder_manager.temp / "model")


def hasAnnotations(dataset: ComputerVisionDataset, excludedClasses: list[str]) -> bool:
    annotationCount = 0

    for clazz in dataset.classDistribution:
        if clazz not in excludedClasses:
            annotationCount += clazz.count

    return annotationCount > 0


def main(experiment: Experiment[ComputerVisionDataset]):

    if not experiment.parameters["validation"]:
        epochs = experiment.parameters["epochs"]

        experiment.createMetrics([
            Metric.create("loss", "epoch", MetricType.int, "value", MetricType.float, [0, epochs]),
            Metric.create("mAP@0.5", "epoch", MetricType.int, "value", MetricType.float, [0, epochs], [0, 1]),
            Metric.create("mAP@0.5:0.95", "epoch", MetricType.int, "value", MetricType.float, [0, epochs], [0, 1])
        ])

        modelDirPath = folder_manager.createTempFolder("model")

        excludedClasses: list[str] = experiment.parameters["excludedClasses"]
        logging.info(f">> [Workspace] Excluding classes: {excludedClasses}")
        experiment.dataset.classes.exclude(excludedClasses)

        experiment.dataset.download(ignoreCache=True)

        if not hasAnnotations(experiment.dataset, excludedClasses):
            raise RuntimeError(">> [Object Detection] No annotations found in the provided dataset. Please add at least 1 annotation before training an object detector.")

        augmentationDataset = experiment.parameters.get("augmentationDataset", None)
        if augmentationDataset is not None:
            logging.info(">> [Object Detection] Augmenting the dataset")

            rotationAngle = experiment.parameters.get("rotationAngle", 0)
            scaleFactor = experiment.parameters.get("scaleFactor", 1.0)

            augmentationDataset.download()

            augmentDataset(experiment.dataset, augmentationDataset, rotationAngle, scaleFactor)

        # start the training
        f1 = train.main(experiment)

        imageSize = experiment.parameters["imageSize"]
        detect.run(experiment, imgsz = (imageSize, imageSize))

        accuracy = f1.mean() if isinstance(f1, np.ndarray) else f1

        export.run(weights = "./weights/best.pt", include = ("tfjs", "coreml"))
        export.run(weights = "./weights/best.pt", include = ("tflite",))  # tfjs, and tflite format need to be exported separately

        prepareModelForUpload("./weights/best.pt",          "./weights/model.pt")
        prepareModelForUpload("./weights/best_web_model",   "./weights/tensorflowjs-model")
        prepareModelForUpload("./weights/best.mlmodel",     "./weights/model.mlmodel")
        prepareModelForUpload("./weights/best-fp16.tflite", "./weights/model.tflite")

        labels = [
            {
                "label": clazz.label,
                "color": clazz.color
            }
            for clazz in experiment.dataset.classes
        ]

        Model.saveModelDescriptor(modelDirPath, {
            "project_task": experiment.spaceTask,
            "labels": labels,
            "modelName": experiment.name,
            "description": experiment.description,

            "input_description": """
                Normalized (values of pixels in range: 0 - 1) RGB image
                Training (Torch model) input shape is [batchSize, channels, height, width]
                Exported (Tensorflow / JS model) input shape is [batchSize, height, width, channels]

                None/Null/Nil values mark dynamic (variable) amount of values
            """,
            "input_shape": [None, imageSize, imageSize, 3],

            "output_description": """
                Output is an array of 4 different values:
                Index 0: Bounding boxes
                Index 1: Scores (range: 0 - 1) - how confident model is for the detected class
                Index 2: Class of the detection
                Index 3: Number of (valid) detections after performing NMS (Non-maximum suppression)

                None/Null/Nil values mark dynamic (variable) amount of values
            """,
            "output_shape": [
                [1, None, 4],
                [1, None],
                [1, None],
                [1]
            ]
        })

        coretexModel = Model.createModel(experiment.name, experiment.id, accuracy, {})
        coretexModel.upload(modelDirPath)

    else:
        imageSize = experiment.parameters["imageSize"]
        modelId = experiment.parameters["modelId"]
        model = Model.fetchById(modelId)

        if model is None:
            raise RuntimeError(f">> [ObjectDetection] Failed to fetch model with provided id: {modelId}")

        model.download()
        weightsPath = Path(folder_manager.modelsFolder) / f"{modelId}/model.pt"

        detect.run(experiment, imgsz = (imageSize, imageSize), weights = weightsPath)


if __name__ == "__main__":
    initializeProject(main)
