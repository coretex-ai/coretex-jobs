from typing import Optional
from pathlib import Path

import json
import random
import logging

from coretex import currentTaskRun, TaskRun, ImageDataset, ImageSample,\
    folder_manager, ImageDatasetClasses, CoretexImageAnnotation, Model
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

import yaml

from src import callback as cb, predict


def loadAnnotation(sample: ImageSample) -> Optional[CoretexImageAnnotation]:
    if not sample.annotationPath.exists():
        return None

    with sample.annotationPath.open("r") as file:
        return CoretexImageAnnotation.decode(json.load(file))


def prepareSample(sample: ImageSample, classes: ImageDatasetClasses, destination: Path) -> None:
    sample.unzip()

    # Create a hard link of image instead of copy
    sample.imagePath.link_to(destination / f"{sample.id}{sample.imagePath.suffix}")

    annotation = loadAnnotation(sample)
    if annotation is not None:
        # YOLO annotation format
        # 1. class
        # 2. bbox - normalized center x
        # 3. bbox - normalized center y
        # 4. bbox - normalized width
        # 5. bbox - normalized height
        normalizedAnnotations: list[tuple[int, float, float, float, float]] = []

        for instance in annotation.instances:
            labelId = classes.labelIdForClassId(instance.classId)

            # Orphan annotation instance - no matching class in dataset
            if labelId is None:
                continue

            box = instance.bbox

            centerX = (box.minX + box.width / 2) / annotation.width
            centerY = (box.minY + box.height / 2) / annotation.height
            width = box.width / annotation.width
            height = box.height / annotation.height

            normalizedAnnotations.append((labelId, centerX, centerY, width, height))

        yoloAnnotationPath = destination / f"{sample.id}.txt"
        with yoloAnnotationPath.open("w") as file:
            for labelId, centerX, centerY, width, height in normalizedAnnotations:
                file.write(f"{labelId} {centerX} {centerY} {width} {height}\n")


def prepareDataset(dataset: ImageDataset, destination: Path, validationPct: float) -> tuple[Path, Path]:
    validCount = int(dataset.count * validationPct)
    trainCount = dataset.count - validCount

    samples = dataset.samples.copy()
    random.shuffle(samples)

    trainPath = destination / "train"
    trainPath.mkdir()

    validPath = destination / "valid"
    validPath.mkdir()

    trainSamples = samples[:trainCount]
    validSamples = samples[trainCount:]

    for sample in trainSamples:
        prepareSample(sample, dataset.classes, trainPath)

    for sample in validSamples:
        prepareSample(sample, dataset.classes, validPath)

    return trainPath, validPath


def generateDatasetYaml(
    datasetPath: Path,
    trainPath: Path,
    validPath: Path,
    configurationPath: Path,
    classes: ImageDatasetClasses
) -> Path:

    classesCategorical: dict[int, str] = {}

    for clazz in classes:
        labelId = classes.labelIdForClass(clazz)
        if labelId is None:
            continue

        classesCategorical[labelId] = clazz.label

    classesCategorical = dict(sorted(classesCategorical.items()))

    configuration = {
        "path": str(datasetPath.absolute()),
        "train": str(trainPath.relative_to(datasetPath)),
        "val": str(validPath.relative_to(datasetPath)),
        "names": classesCategorical
    }

    with configurationPath.open("w") as file:
        yaml.safe_dump(configuration, file, sort_keys = False)

    return configurationPath


def calculateF1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0

    return 2 * ((precision * recall) / (precision + recall))


def getPatience(taskRun: TaskRun) -> int:
    epochs = taskRun.parameters["epochs"]
    earlyStopping = taskRun.parameters["earlyStopping"]

    if earlyStopping:
        # 10% of epochs or 10, whichever is higher
        return max(int(epochs * 0.1), 10)

    # To disable early stopping we have to set patience to a very high value
    # https://github.com/ultralytics/ultralytics/issues/7609
    return 2 ** 64


def isValidationSplitValid(validationSplit: float, datasetSize: int) -> bool:
    if not 0 <= validationSplit < 1:
        logging.error(f">> [ObjectDetection] validationSplit parameter ({validationSplit}) must be between 0 and 1")
        return False

    minSamplesForSplit = int(1 / min(validationSplit, 1 - validationSplit))
    if datasetSize < minSamplesForSplit:
        logging.error(f">> [ObjectDetection] Dataset is too small ({datasetSize}) for validationSplit parameter ({validationSplit}). Minimum number of samples is {minSamplesForSplit}")
        return False

    return True


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    if not isValidationSplitValid(taskRun.parameters["validationSplit"], taskRun.dataset.count):
        raise ValueError(f">> [ObjectDetection] validationSplit parameter is invalid")

    taskRun.dataset.download()
    taskRun.dataset.classes.exclude(taskRun.parameters["excludedClasses"])

    datasetPath = folder_manager.createTempFolder("dataset")
    trainPath, validPath = prepareDataset(taskRun.dataset, datasetPath, taskRun.parameters["validationSplit"])

    configurationPath = folder_manager.temp / "dataset_configuration.yaml"
    generateDatasetYaml(datasetPath, trainPath, validPath, configurationPath, taskRun.dataset.classes)

    ctxModel: Optional[Model] = taskRun.parameters.get("model")
    if ctxModel is None:
        # Start training from specified YoloV8 weights
        weights = taskRun.parameters.get("weights", "yolov8n.pt")
        logging.info(f">> [ObjectDetection] Using \"{weights}\" for training the model")

        model = YOLO(taskRun.parameters.get("weights", "yolov8n.pt"))
    else:
        logging.info(f">> [ObjectDetection] Using \"{ctxModel.name}\" for training the model")

        # Start training from specified model checkpoint
        ctxModel.download()
        model = YOLO(ctxModel.path / "best.pt")

    model.add_callback("on_train_start", cb.onTrainStart)
    model.add_callback("on_train_epoch_end", cb.onEpochEnd)

    logging.info(">> [ObjectDetection] Training the model")
    results: DetMetrics = model.train(
        project = "results",
        data = configurationPath,
        epochs = taskRun.parameters["epochs"],
        batch = taskRun.parameters["batchSize"],
        imgsz = taskRun.parameters["imageSize"],
        patience = getPatience(taskRun)
    )

    precision = results.results_dict["metrics/precision(B)"]
    recall = results.results_dict["metrics/recall(B)"]
    f1 = calculateF1(precision, recall)

    ctxModel = Model.createModel(taskRun.name, taskRun.id, f1, {})
    ctxModel.upload(Path(".", "results", "train", "weights"))

    logging.info(">> [ObjectDetection] Running prediction on training dataset")
    predictPath = folder_manager.createTempFolder("predict")
    predict.run(model, taskRun.dataset, predictPath, taskRun.parameters["batchSize"])

    for path in predictPath.iterdir():
        if not path.is_file():
            continue

        taskRun.createArtifact(path, path.name)


if __name__ == "__main__":
    main()
