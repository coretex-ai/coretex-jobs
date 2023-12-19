from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass

import logging
import csv

from coretex import currentTaskRun, folder_manager, ComputerVisionDataset, TaskRun, BBox, ComputerVisionSample
from coretex.utils import mathematicalRound
from ultralytics import YOLO

from src import detect_document, document_extractor, model
from src.image_segmentation import processMask, segmentDetections
from src.ocr import performOCR
from src.utils import savePlot, saveDocumentWithDetections
from src.object_detection import runObjectDetection


@dataclass
class SampleAccuracyResult:

    id: int
    name: str
    labelAccuracies: dict[str, float]

    @property
    def accuracy(self) -> float:
        return mathematicalRound(sum(self.labelAccuracies.values()) / len(self.labelAccuracies), 2)


def calculateAccuracy(sample: ComputerVisionSample, groundtruth: dict[str, BBox], prediction: dict[str, BBox]) -> SampleAccuracyResult:
    labelAccuracies: dict[str, float] = {}

    for label, groundtruthBBox in groundtruth.items():
        predictionBBox = prediction.get(label)

        if predictionBBox is None:
            labelAccuracies[label] = 0.0
        else:
            labelAccuracies[label] = mathematicalRound(groundtruthBBox.iou(predictionBBox) * 100, 2)

    return SampleAccuracyResult(sample.id, sample.name, labelAccuracies)


def processSample(
    taskRun: TaskRun[ComputerVisionDataset],
    sample: ComputerVisionSample,
    segmentationModel: Any,
    detectionModel: YOLO
) -> Optional[SampleAccuracyResult]:

    sample.unzip()

    predictedMask = detect_document.run(segmentationModel, sample)
    mask = processMask(predictedMask)

    if mask is None:
        return None

    data = sample.load()
    segmentedImage, transformedAnnotation = document_extractor.extractDocumentImage(
        data.image,
        mask,
        taskRun.dataset.classes,
        data.annotation
    )

    savePlot(sample, predictedMask, mask, taskRun)

    bboxes, classes = runObjectDetection(segmentedImage, detectionModel)

    sampleOutputdir = folder_manager.createTempFolder(sample.name)
    segmentedDetections = segmentDetections(segmentedImage, bboxes, classes, sampleOutputdir, taskRun)
    saveDocumentWithDetections(segmentedImage, bboxes, classes, sampleOutputdir, taskRun)
    performOCR(segmentedDetections, classes, sampleOutputdir, taskRun)

    if transformedAnnotation is not None:
        logging.info("\tSample has annotations, calculating accuracy")

        prediction = {label: bbox for label, bbox in zip(classes, bboxes)}
        groundtruth: dict[str, BBox] = {}

        for instance in transformedAnnotation.instances:
            class_ = taskRun.dataset.classes.classById(instance.classId)
            if class_ is None or class_.label.startswith("document"):
                continue

            groundtruth[class_.label] = instance.bbox

        sampleResult = calculateAccuracy(sample, groundtruth, prediction)

        logging.info(f"\tTotal accuracy: {sampleResult.accuracy}")
        logging.info(f"\tAccuracy per label:")

        for label, accuracy in sampleResult.labelAccuracies.items():
            logging.info(f"\t- {label}: {accuracy}")

        return sampleResult

    return None


def generateResultsCsv(sampleResults: list[SampleAccuracyResult], destination: Path) -> None:
    with destination.open("w") as file:
        writer = csv.DictWriter(file, ["id", "name", "first_name", "last_name", "date_of_birth", "gender", "total"])
        writer.writeheader()
        for result in sampleResults:
            writer.writerow({
                "id": result.id,
                "name": result.name,
                "first_name": result.labelAccuracies.get("first_name", "-"),
                "last_name": result.labelAccuracies.get("last_name", "-"),
                "date_of_birth": result.labelAccuracies.get("date_of_birth", "-"),
                "gender": result.labelAccuracies.get("gender", "-"),
                "total": result.accuracy
            })


def main() -> None:
    taskRun: TaskRun[ComputerVisionDataset] = currentTaskRun()
    taskRun.dataset.download()

    segmentationModel = model.loadSegmentationModel(taskRun.parameters["segmentationModel"])
    detectionModel = model.loadDetectionModel(taskRun.parameters["objectDetectionModel"])

    sampleResults: list[SampleAccuracyResult] = []

    for i, sample in enumerate(taskRun.dataset.samples):
        logging.info(f">> [Document OCR] Performing segmentation on sample \"{sample.name}\" ({i + 1}/{taskRun.dataset.count})")

        sampleResult = processSample(taskRun, sample, segmentationModel, detectionModel)
        if sampleResult is not None:
            sampleResults.append(sampleResult)

    datasetResult: dict[str, float] = {}

    for sampleResult in sampleResults:
        for label, accuracy in sampleResult.labelAccuracies.items():
            totalAccuracy = datasetResult.get(label, 0)
            datasetResult[label] = totalAccuracy + accuracy

    for key, value in datasetResult.items():
        datasetResult[key] = value / len(sampleResults)

    datasetAccuracy = sum(datasetResult.values()) / len(datasetResult)
    logging.info(f">> [Document OCR] Dataset accuracy: {datasetAccuracy}")

    resultsPath = folder_manager.temp / "results.csv"
    generateResultsCsv(sampleResults, resultsPath)

    resultsArtifact = taskRun.createArtifact(resultsPath, "results.csv")
    if resultsArtifact is None:
        logging.error(">> [Document OCR] Failed to create results artifact")


if __name__ == "__main__":
    main()
