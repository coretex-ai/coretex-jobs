from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from pathlib import Path

import logging
import csv
import multiprocessing as mp

from PIL import Image, ImageDraw
from coretex import TaskRun, ImageDataset, ImageDatasetClasses, folder_manager, TaskRunStatus, ImageSample
from ultralytics import YOLO
from matplotlib import pyplot as plt
from ultralytics.engine.results import Results, Masks
from coretex.networking.network_manager import networkManager

import numpy as np
import torch


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


def uploadSampleArtifacts(taskRun: TaskRun[ImageDataset], sampleId: int, plotPath: Path, sampleResult: dict[str, str]) -> None:
    csvSamplePath = folder_manager.temp.joinpath(f"{sampleId}.csv")
    with csvSamplePath.open("w") as file:
        writer = csv.DictWriter(file, fieldnames = sampleResult.keys())
        writer.writeheader()
        writer.writerow(sampleResult)

    if taskRun.createArtifact(csvSamplePath, f"sample_results/{sampleId}/results.csv") is None:
        logging.warning(f">> [Image Segmentation] Failed to upload csv file with results for sample {sampleId} as artifact")

    if taskRun.createArtifact(plotPath, f"sample_results/{sampleId}/{plotPath.name}") is None:
        logging.warning(f">> [Image Segmentation] Failed to upload image \"{plotPath.name}\" with segmentation as artifact")


def plotSegmentationImage(
    image: np.ndarray,
    sampleId: int,
    origSeg: np.ndarray,
    predictedMask: np.ndarray,
    iou: float
) -> Path:

    fig, axes = plt.subplots(1, 3, figsize = (20, 10))

    ax1 = axes[0]
    ax1.imshow(image, cmap = "summer")
    ax1.set_title(f"Original image, id: {sampleId}")
    ax1.axis("off")

    ax2 = axes[1]
    ax2.imshow(origSeg, cmap = "summer")
    ax2.set_title(f"Original segmentation")
    ax2.axis("off")

    ax3 = axes[2]
    ax3.imshow(predictedMask, cmap = "summer")
    ax3.set_title(f"Predicted segmentation\nAcc: {iou:.2f}")
    ax3.axis("off")

    plotPath = folder_manager.temp.joinpath(f"{sampleId}.jpeg")
    plt.savefig(plotPath)
    fig.clear()
    plt.close()

    return plotPath


def iouScoreClass(testMask: np.ndarray, predictedMask: np.ndarray) -> float:
    intersecrion = np.logical_and(testMask, predictedMask)
    union = np.logical_or(testMask, predictedMask)

    return float(np.sum(intersecrion) / np.sum(union) * 100)


def maskFromPoly(polygons: list[list[tuple[int, int]]], width: int, height: int) -> np.ndarray:
    image = Image.new("L", (width, height))
    for polygon in polygons:
        draw = ImageDraw.Draw(image)
        draw.polygon(polygon, fill = 1)

    return np.array(image)


def calculateSingleClassAccuracy(
    classId: int,
    classIds: list[int],
    polygons: list[list[tuple[int, int]]],
    width: int,
    height: int,
    origClassSeg: np.ndarray
) -> float:

    indexes = [i for i, clsId in enumerate(classIds) if clsId == classId]
    classPolygons = [polygons[i] for i in indexes]
    predClassMask = maskFromPoly(classPolygons, width, height)
    iouClass = iouScoreClass(origClassSeg, predClassMask)

    return iouClass


def extractPolygonsFromResultsMasks(masks: Masks, height: int, width: int) -> list[list[tuple[int, int]]]:
    polygons: list[list[tuple[int, int]]] = []
    for mask in masks:
        polygon = mask.xyn
        for points in polygon:
            poly = points.tolist()
            poly = [tuple(point) for point in poly]
            poly = [(int(x * width), int(y * height)) for (x, y) in poly]
            polygons.append(poly)

    return polygons


def processSampleResult(
    taskRun: TaskRun[ImageDataset],
    result: Results,
    sample: ImageSample,
    refreshedToken: str
) -> dict[str, str]:

    networkManager.authenticateWithRefreshToken(refreshedToken)

    sampleAnnotation = sample.load().annotation

    csvRowResult: dict[str, str] = {}
    csvRowResult["sample id"] = f"{sample.id}"
    csvRowResult["sample name"] = sample.name
    for className in taskRun.dataset.classes.labels:
        csvRowResult[className] = "0.00"

    height = result.orig_shape[0]
    width = result.orig_shape[1]
    if result.masks is not None:
        polygons = extractPolygonsFromResultsMasks(result.masks, height, width)

        classIds = [int(x) for x in result.boxes.cls.tolist()]
        classNames: dict[int, str] = result.names

        for classId in list(set(classIds)):
            clazz = taskRun.dataset.classByName(classNames[classId])
            if clazz is not None:
                if sampleAnnotation is not None:
                    groundtruthClassMask = sampleAnnotation.extractSegmentationMask(ImageDatasetClasses([clazz]))
                else:
                    groundtruthClassMask = np.zeros((height, width))
                csvRowResult[classNames[classId]] = f"{calculateSingleClassAccuracy(classId, classIds, polygons, width, height, groundtruthClassMask):.2f}"
            else:
                csvRowResult[classNames[classId]] = "0.00"

        predMask = maskFromPoly(polygons, width, height)
    else:
        predMask = np.zeros((height, width))

    if sampleAnnotation is not None:
        groundtruthMask = sampleAnnotation.extractSegmentationMask(taskRun.dataset.classes)
    else:
        groundtruthMask = np.zeros((height, width))

    iou = iouScoreClass(groundtruthMask, predMask)
    csvRowResult["accuracy"] = f"{iou:.2f}"

    plotPath = plotSegmentationImage(result.orig_img, sample.id, groundtruthMask, predMask, iou)
    uploadSampleArtifacts(taskRun, sample.id, plotPath, csvRowResult)

    return csvRowResult


def validate(taskRun: TaskRun[ImageDataset], modelPath: Path, imgSize: int) -> float:
    taskRun.updateStatus(TaskRunStatus.inProgress, "Validating")

    model = YOLO(modelPath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f">> [Image Segmentation] The {str(device).upper()} will be used for validating")

    samples = taskRun.dataset.samples
    batchSize = taskRun.parameters["batchSize"]

    logging.info(f">> [Image Segmentation] Validating the model with batch size {batchSize}")

    processedResults: list[dict[str, str]] = []
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers = mp.cpu_count(), mp_context = context) as executor:
        counter = 1
        for startIndex in range(0, taskRun.dataset.count, batchSize):
            batchSamples = samples[startIndex:startIndex + batchSize]
            batchSamplesPaths = [sample.imagePath for sample in batchSamples]

            results = model.predict(
                source = batchSamplesPaths,
                save = False,
                imgsz = imgSize,
                plots = False,
                conf = taskRun.parameters["confidenceTreshold"]
            )

            futures: list[Future[dict[str, str]]] = []

            refreshedToken = networkManager._refreshToken
            if refreshedToken is None:
                raise RuntimeError(">> [Image Segmentation] Refreshing the authentication token failed.")

            for result, sample in zip(results, batchSamples):
                future = executor.submit(processSampleResult, taskRun, result, sample, refreshedToken)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    processedSampleResult = future.result()
                    processedResults.append(processedSampleResult)
                    logging.info(f"Postprocessing for sample {counter}/{taskRun.dataset.count} is finished. Sample ID: {processedSampleResult['sample id']} -> accuracy: {processedSampleResult['accuracy']}")
                    counter += 1
                except FileNotFoundError as e:
                    logging.warning(e)

    taskRun.updateStatus(TaskRunStatus.inProgress, "Creating csv fils with results")
    createSampleResults(taskRun, processedResults)
    createDatasetResults(taskRun, processedResults)

    datasetAcc = [float(result['accuracy']) for result in processedResults]
    accuracy = float(np.mean(datasetAcc))
    logging.info(f">> [Image Segmentation] Dataset accuracy is: {accuracy}")

    return accuracy
