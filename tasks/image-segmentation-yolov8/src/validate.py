from pathlib import Path

import json
import csv
import logging

from ultralytics import YOLO
from PIL import Image
from pycocotools import mask as maskCoco
from matplotlib import pyplot as plt
from coretex import TaskRun, ImageDataset, folder_manager, Model, ImageSample, ImageDatasetClasses

import numpy as np

from .dataset import prepareDataset, createYamlFile


def fetchCsvDatasetAcc(csvSamplesPath: Path) -> list[float]:
    with csvSamplesPath.open("r") as file:
        reader = csv.DictReader(file)
        csvDatasetAcc = [float(row["accuracy"]) for row in reader]

    return csvDatasetAcc


def fetchCsvClassesData(labels: list[str], csvSamplesPath: Path) -> dict[str, list[float]]:
    csvClassesData: dict[str, list[float]] = {}

    with csvSamplesPath.open("r") as file:
        reader = csv.DictReader(file)
        for label in labels:
            classAcc = [float(row[label]) for row in reader]
            csvClassesData[label] = classAcc

    return csvClassesData


def generateFieldNamesDataset(labels: list[str]) -> list[str]:
    fieldNamesDataset: list[str] = []

    for label in labels:
        fieldNamesDataset.append(f"{label} acc")
        fieldNamesDataset.append(f'{label} acc STD')

    fieldNamesDataset.extend(["accuracy", "accuracy STD"])

    return fieldNamesDataset


def createCsvDatasetResults(taskRun: TaskRun[ImageDataset], csvSamplesPath: Path) -> float:
    fieldNamesDataset = generateFieldNamesDataset(taskRun.dataset.classes.labels)

    csvDatasetPath = folder_manager.createTempFolder("csv_dataset") / "dataset_results.csv"

    csvClassesData = fetchCsvClassesData(taskRun.dataset.classes.labels, csvSamplesPath)
    csvDatasetAcc = fetchCsvDatasetAcc(csvSamplesPath)

    csvDatasetResults: dict[str, str] = {}
    for label in taskRun.dataset.classes.labels:
        csvDatasetResults[f"{label} acc"] = f"{np.mean(csvClassesData[label]):.2f}"
        csvDatasetResults[f"{label} acc STD"] = f"{np.std(csvClassesData[label]):.2f}"

    csvDatasetResults["accuracy"] = f"{np.mean(csvDatasetAcc):.2f}"
    csvDatasetResults["accuracy STD"] = f"{np.std(csvDatasetAcc):.2f}"

    with csvDatasetPath.open("w") as file:
        writer = csv.DictWriter(file, fieldnames = fieldNamesDataset)
        writer.writeheader()
        writer.writerow(csvDatasetResults)

    if taskRun.createArtifact(csvDatasetPath, csvDatasetPath.name) is None:
        logging.warning(">> [Image Segmentation] Failed to upload csv file with dataset results as artifact")
    else:
        logging.info(">> [Image Segmentation] The csv file with dataset results has been uploaded as artifact")

    return float(csvDatasetResults["accuracy"])


def plotSegmentationImage(
    sample: ImageSample,
    origSeg: np.ndarray,
    predictedMask: np.ndarray,
    sampleAcc: float,
    artifactImagesPath: Path
) -> None:

    img = Image.open(sample.imagePath)

    fig, axes = plt.subplots(1, 3, figsize = (20, 10))

    axes[0].imshow(img, cmap = "summer")
    axes[0].set_title(f"Original image, id: {sample.id}")
    axes[0].axis("off")

    axes[1].imshow(origSeg, cmap = "summer")
    axes[1].set_title(f"Original segmentation")
    axes[1].axis("off")

    axes[2].imshow(predictedMask, cmap = "summer")
    axes[2].set_title(f"Predicted segmentation\nAcc: {sampleAcc:.2f}")
    axes[2].axis("off")

    plt.savefig(artifactImagesPath / f"{sample.id}.jpeg")
    plt.close()


def iouScoreClass(testMask: np.ndarray, predictedMask: np.ndarray) -> float:
    intersecrion = np.logical_and(testMask, predictedMask)
    union = np.logical_or(testMask, predictedMask)

    return float(np.sum(intersecrion) / np.sum(union))


def processPrediction(
    taskRun: TaskRun[ImageDataset],
    predictData: list[dict],
    csvSamplesPath: Path
) -> None:

    artifactImagesPath = folder_manager.createTempFolder("images")

    fieldNamesSamples = ["sample id", "sample name"]
    fieldNamesSamples.extend(taskRun.dataset.classes.labels)
    fieldNamesSamples.append("accuracy")

    with csvSamplesPath.open("w") as file:
        writer = csv.DictWriter(file, fieldnames = fieldNamesSamples)
        writer.writeheader()

    for sample in taskRun.dataset.samples:
        img = Image.open(sample.imagePath)
        origSeg = sample.load().annotation.extractSegmentationMask(taskRun.dataset.classes)

        predictedMask = np.zeros((img.height, img.width))

        csvSampleValues = [str(sample.id), sample.name]

        iouClasses: list[float] = []
        for classId, clazz in enumerate(taskRun.dataset.classes):
            predSampleData = [data for data in predictData if data["image_id"] == sample.id and data["category_id"] == classId]

            if len(predSampleData) <= 0:
                logging.info(f">> [Image Segmentation] The sample \"{sample.name}\" (sample id: {sample.id}) does not have segmentation for the class \"{clazz.label}\"")
                continue

            predClassSegmentation = predSampleData[0]

            predClassMask = predClassSegmentation["segmentation"]
            decodedClassMask = maskCoco.decode(predClassMask)
            predictedMask = np.logical_or(predictedMask, decodedClassMask).astype(int)

            iouClass = iouScoreClass(sample.load().annotation.extractSegmentationMask(ImageDatasetClasses([clazz])), decodedClassMask)
            iouClasses.append(iouClass)
            csvSampleValues.append(f"{iouClass * 100:.2f}")

        if len(iouClasses) > 0:
            sampleAcc = sum(iouClasses) / len(iouClasses) * 100
            csvSampleValues.append(f"{sampleAcc:.2f}")

            with csvSamplesPath.open("a") as file:
                writer = csv.DictWriter(file, fieldnames = fieldNamesSamples)
                writer.writerow(dict(zip(fieldNamesSamples, csvSampleValues)))

            plotSegmentationImage(sample, origSeg, predictedMask, sampleAcc, artifactImagesPath)

    for path in artifactImagesPath.iterdir():
        if path.is_file():
            if taskRun.createArtifact(path, f"images/{path.name}") is None:
                logging.warning(f">> [Image Segmentation] Failed to upload image \"{path.name}\" with segmentation as artifact")
            else:
                logging.info(f">> [Image Segmentation] The segmentation image \"{path.name}\" has been uploaded as artifact")

    if taskRun.createArtifact(csvSamplesPath, csvSamplesPath.name) is None:
        logging.warning(">> [Image Segmentation] Failed to upload csv file with samples results as artifact")
    else:
        logging.info(">> [Image Segmentation] The csv file with samples results has been uploaded as artifact")

def validate(taskRun: TaskRun[ImageDataset]) -> float:
    datasetPath = folder_manager.createTempFolder("dataset_val")
    trainDatasetPath, validDatasetPath = prepareDataset(taskRun.dataset, datasetPath, 1.0)

    yamlFilePath = datasetPath / "config.yaml"
    createYamlFile(datasetPath, trainDatasetPath, validDatasetPath, taskRun.dataset.classes, yamlFilePath)

    ctxModel: Model = taskRun.parameters["trainedModel"]
    ctxModel.download()
    model = YOLO(ctxModel.path / "best.pt")

    valResPath = folder_manager.createTempFolder("validation_results")

    logging.info(">> [Image Segmentation] Validating the model")
    model.val(
        project = valResPath,
        data = yamlFilePath,
        batch = taskRun.parameters["batchSize"],
        imgsz = taskRun.parameters["imageSize"],
        save_json = True,
    )

    jsonFiles = valResPath.rglob("predictions.json")
    for path in jsonFiles:
        jsonPath = path

    with jsonPath.open("r") as file:
        predictData = list(json.load(file))

    csvSamplesPath = folder_manager.createTempFolder("csv_samples") / "sample_results.csv"

    processPrediction(taskRun, predictData, csvSamplesPath)

    accuracy = createCsvDatasetResults(taskRun, csvSamplesPath)

    return accuracy
