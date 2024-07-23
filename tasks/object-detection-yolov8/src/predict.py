from typing import Optional
from pathlib import Path

import logging
import csv
import math

from coretex import TaskRun, folder_manager, ImageDataset, ImageDatasetClasses, ImageDatasetClass, BBox, ImageSample, BBox, CoretexSegmentationInstance
from ultralytics import YOLO
from ultralytics.engine.results import Results

import matplotlib.pyplot as plt
import matplotlib.patches as pth


# Minimum number of values in an annotation instance [x1, y1, x2, y2...] for it to be a two dimentional object
DIMENSION_THRESHOLD = 6


def std(numbers: list[float]) -> float:
    mean = sum(numbers) / len(numbers)
    sqDif = [(x - mean) ** 2 for x in numbers]
    meanSqDif = sum(sqDif) / len(sqDif)

    return math.sqrt(meanSqDif)


def classByLabelId(labelId: int, classes: ImageDatasetClasses) -> Optional[ImageDatasetClass]:
    return classes.classByLabel(classes.labels[labelId])


def iouScore(
    result: Results,
    sample: ImageSample,
    classes: ImageDatasetClasses,
    treshold: float,
    csvSamplesData: list[dict]
) -> float:

    csvSampleData: dict[str, str] = {}
    csvSampleData["Sample ID"] = str(sample.id)
    csvSampleData["Sample Name"] = str(sample.name)

    for className in [clazz.label for clazz in classes]:
        csvSampleData[className] = "0"

    iouSample: list[float] = []

    if result.boxes is not None:
        instances = sample.load().annotation.instances
        classesNames = [classes.classById(instance.classId) for instance in instances if classes.classById(instance.classId) is not None]

        for minX, minY, maxX, maxY, confidence, labelId in result.boxes.data:
            clazz = classByLabelId(int(labelId), classes)
            if clazz is None:
                continue



            for instance in instances:
                if instance.classId in clazz.classIds:




        """
        for minX, minY, maxX, maxY, confidence, labelId in result.boxes.data:
            clazz = classByLabelId(int(labelId), classes)
            if clazz is None:
                continue

            predictedBBox = BBox.create(minX, minY, maxX, maxY)

            for instance in instances:
                if instance.classId in clazz.classIds:
                    iouBBox = instance.bbox.iou(predictedBBox)
                    iouSample.append(float(iouBBox))
                    csvSampleData[clazz.label] = str(round(float(iouBBox), 2))
        """
        try:
            print(iouSample)
            iouS = sum(iouSample) / len(classesNames) if len(classesNames) > len(iouSample) else sum(iouSample) / len(iouSample)
            #klase = [classes.classById(instance.classId) for instance in instances]
            #logging.warning(klase)
            #logging.warning(f"broj klasa je: {len(klase)}")
            #klase = [klasa for klasa in klase if klasa is not None]
            #logging.warning(klase)
            #logging.warning(f"broj klasa posle je: {len(klase)}")
            #logging.warning([klasa.label for klasa in klase])
            logging.warning(f"Ime: {sample.name} --> Broj klasa: {len(classesNames)} --> IoU: {iouS}")
            csvSampleData["IoU Score"] = str(round(iouS, 2))
            csvSampleData["Accuracy"] = str(100 if iouS > treshold else 0)
            csvSamplesData.append(csvSampleData)
            return float(iouS)
        except ZeroDivisionError:
            csvSampleData["IoU Score"] = "0"
            csvSampleData["Accuracy"] = "0"
            csvSamplesData.append(csvSampleData)
            return 0.0
    else:
        csvSampleData["IoU Score"] = "0"
        csvSampleData["Accuracy"] = "0"
        csvSamplesData.append(csvSampleData)
        return 0.0


def processResult(
    result: Results,
    classes: ImageDatasetClasses,
    sample: ImageSample,
    iou: float,
    savePath: Path
) -> None:

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 10))

    ax1.imshow(result.orig_img)
    ax1.set_title("Original")
    ax1.axis("off")

    for instance in sample.load().annotation.instances:
        clazz = classes.classById(instance.classId)
        if clazz is not None:
            ax1.add_patch(pth.Rectangle(
                    (float(instance.bbox.minX), float(instance.bbox.minY)),
                    float(instance.bbox.width),
                    float(instance.bbox.height),
                    linewidth = 3,
                    edgecolor = clazz.color,
                    facecolor = "none"
                ))

    ax2.imshow(result.orig_img)
    ax2.set_title(f"Predicted\nIoU Score: {round(iou, 2)}")
    ax2.axis("off")

    if result.boxes is not None:
        for minX, minY, maxX, maxY, confidence, labelId in result.boxes.data:
            box = BBox.create(minX, minY, maxX, maxY)

            clazz = classByLabelId(int(labelId), classes)
            if clazz is None:
                continue

            ax2.add_patch(pth.Rectangle(
                (float(box.minX), float(box.minY)),
                float(box.width),
                float(box.height),
                linewidth = 3,
                edgecolor = clazz.color,
                facecolor = "none"
            ))

    plt.savefig(savePath)
    plt.close()

def isSampleValid(sample: ImageSample) -> bool:
    try:
        instances = sample.load().annotation.instances
        if instances is None:
            return False

        for instance in instances:
            if any(len(segmentation) < DIMENSION_THRESHOLD for segmentation in instance.segmentations):
                return False
    except Exception as e:
        logging.debug(f"Falied to load sample annotation data for {sample.name}, ID: {sample.id}. Error: {e}")
        return False

    return True


def predictBatch(
    model: YOLO,
    dataset: ImageDataset,
    startIdx: int,
    endIdx: int,
    treshold: float,
    resultPath: Path,
    csvSamplesData: list[dict]
) -> None:

    batch = [sample for sample in dataset.samples[startIdx:endIdx] if isSampleValid(sample)]
    results: Results = model.predict([sample.imagePath for sample in batch])
    for sample, result in zip(batch, results):
        iou = iouScore(result, sample, dataset.classes, treshold, csvSamplesData)
        processResult(result, dataset.classes, sample, iou, resultPath / f"{sample.name}.png")


def run(
    taskRun: TaskRun,
    model: YOLO,
    dataset: ImageDataset,
    resultPath: Path,
    batchSize: int
) -> float:

    fieldNamesSamples = ["Sample ID", "Sample Name"]
    fieldNamesSamples.extend([clazz.label for clazz in dataset.classes])
    fieldNamesSamples.extend(["IoU Score", "Accuracy"])

    treshold = taskRun.parameters["treshold"]
    if treshold < 0 or treshold > 1:
        raise ValueError(f"The value of the threshold parameter must be between 0 and 1. The current value is: {treshold}")

    csvSamplesData: list[dict[str, str]] = []

    for i in range(0, dataset.count, batchSize):
        predictBatch(model, dataset, i, i + batchSize, treshold, resultPath, csvSamplesData)









    sampleResultsPath = folder_manager.temp / "sample_results.csv"
    with open(sampleResultsPath, "w", newline = "") as csvFile:
        writer = csv.DictWriter(csvFile, fieldNamesSamples)
        writer.writeheader()
        writer.writerows(csvSamplesData)

    if taskRun.createArtifact(sampleResultsPath, "sample_results.csv") is None:
        logging.error(f">> [ObjectDetection] Failed to create artifact \"{sampleResultsPath.name}\"")

    fieldNamesDataset: list[str] = []
    for clazz in [clazz.label for clazz in dataset.classes]:
        fieldNamesDataset.append(clazz)
        fieldNamesDataset.append(f"{clazz} STD")

    fieldNamesDataset.extend(["IoU Score", "IoU STD", "Accuracy"])

    csvDatasetData: dict[str, float] = {}

    for item in [clazz.label for clazz in dataset.classes]:
        classData = [float(data[item]) for data in csvSamplesData]
        csvDatasetData[item] = round(sum(classData) / len(classData), 2)
        csvDatasetData[f"{item} STD"] = round(std(classData), 2)

    iouData = [float(data["IoU Score"]) for data in csvSamplesData]
    csvDatasetData["IoU Score"] = round(sum(iouData) / len(iouData), 2)
    csvDatasetData["IoU STD"] = round(std(iouData), 2)

    accData = [float(data["Accuracy"]) for data in csvSamplesData]
    csvDatasetData["Accuracy"] = round(sum(accData) / len(accData), 2)

    datasetResultsPath = folder_manager.temp / "dataset_results.csv"
    with open(datasetResultsPath, "w", newline = "") as csvFile:
        writer = csv.DictWriter(csvFile, fieldNamesDataset)
        writer.writeheader()
        writer.writerow(csvDatasetData)

    if taskRun.createArtifact(datasetResultsPath, "dataset_results.csv") is None:
        logging.error(f">> [ObjectDetection] Failed to create artifact \"{datasetResultsPath.name}\"")

    return csvDatasetData["IoU Score"]
