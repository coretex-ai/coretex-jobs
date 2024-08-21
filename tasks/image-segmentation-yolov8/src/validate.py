from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from pathlib import Path

import logging
import gc
import csv
import multiprocessing as mp

from PIL import Image, ImageDraw
from coretex import TaskRun, ImageDataset, ImageDatasetClasses, folder_manager, TaskRunStatus, ImageSample
from ultralytics import YOLO
from matplotlib import pyplot as plt
from ultralytics.engine.results import Results, Masks

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
    else:
        logging.info(f">> [Image Segmentation] The csv file with results for sample {sampleId} has been uploaded as artifact")

    if taskRun.createArtifact(plotPath, f"sample_results/{sampleId}/{plotPath.name}") is None:
        logging.warning(f">> [Image Segmentation] Failed to upload image \"{plotPath.name}\" with segmentation as artifact")
    else:
        logging.info(f">> [Image Segmentation] The segmentation image \"{plotPath.name}\" has been uploaded as artifact")


def plotSegmentationImage(
    imagePath: Path,
    sampleId: int,
    origSeg: np.ndarray,
    predictedMask: np.ndarray,
    iou: float
) -> Path:

    img = Image.open(imagePath)
    fig, axes = plt.subplots(1, 3, figsize = (20, 10))

    axes[0].imshow(img, cmap = "summer")
    axes[0].set_title(f"Original image, id: {sampleId}")
    axes[0].axis("off")

    axes[1].imshow(origSeg, cmap = "summer")
    axes[1].set_title(f"Original segmentation")
    axes[1].axis("off")

    axes[2].imshow(predictedMask, cmap = "summer")
    axes[2].set_title(f"Predicted segmentation\nAcc: {iou:.2f}")
    axes[2].axis("off")

    plotPath = folder_manager.temp.joinpath(f"{sampleId}.jpeg")
    plt.savefig(plotPath)
    plt.close()
    gc.collect()  # The garbage collector is manually invoked after each plotting to prevent a memory leak

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
    sample: ImageSample
) -> dict[str, str]:

    sampleAnnotation = sample.load().annotation
    if sampleAnnotation is None:
        raise RuntimeError(f">> [Image Segmentation] The sample named \"{sample.name}\" (id: \"{sample.id}\") has no annotation.")

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
                origClassSeg = sampleAnnotation.extractSegmentationMask(ImageDatasetClasses([clazz]))
                csvRowResult[classNames[classId]] = f"{calculateSingleClassAccuracy(classId, classIds, polygons, width, height, origClassSeg):.2f}"
            else:
                csvRowResult[classNames[classId]] = "0.00"

        predMask = maskFromPoly(polygons, width, height)
    else:
        predMask = np.zeros((height, width))

    iou = iouScoreClass(sampleAnnotation.extractSegmentationMask(taskRun.dataset.classes), predMask)
    csvRowResult["accuracy"] = f"{iou:.2f}"

    plotPath = plotSegmentationImage(result.path, sample.id, sampleAnnotation.extractSegmentationMask(taskRun.dataset.classes), predMask, iou)
    uploadSampleArtifacts(taskRun, sample.id, plotPath, csvRowResult)

    return csvRowResult


def validate(taskRun: TaskRun[ImageDataset], modelPath: Path, imgSize: int) -> float:
    taskRun.updateStatus(TaskRunStatus.inProgress, "Validating")

    model = YOLO(modelPath)
    samples = [sample.imagePath for sample in taskRun.dataset.samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f">> [Image Segmentation] The {str(device).upper()} will be used for validating")

    logging.info(">> [Image Segmentation] Validatin the model")
    results = model.predict(
        source = samples,
        save = False,
        batch = taskRun.parameters["batchSize"],
        imgsz = imgSize,
        plots = False,
        conf = taskRun.parameters["confidenceTreshold"]
    )

    taskRun.updateStatus(TaskRunStatus.inProgress, "Processing predictions")
    processedResults: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers = mp.cpu_count()) as executor:
        futures: list[Future[dict[str, str]]] = []
        for result, sample in zip(results, taskRun.dataset.samples):
            future = executor.submit(processSampleResult, taskRun, result, sample)
            futures.append(future)


        for counter, future in enumerate(as_completed(futures)):
            try:
                processedResults.append(future.result())
            except FileNotFoundError as e:
                logging.warning(e)

            logging.info(f">> [Image Segmentation] Processing results for sample {counter + 1}/{taskRun.dataset.count} has been finished")

    taskRun.updateStatus(TaskRunStatus.inProgress, "Creating csv fils with results")
    createSampleResults(taskRun, processedResults)
    createDatasetResults(taskRun, processedResults)

    datasetAcc = [float(result['accuracy']) for result in processedResults]

    return float(np.mean(datasetAcc))
