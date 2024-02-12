from typing import Any, Optional

import logging

from coretex import currentTaskRun, folder_manager, ComputerVisionDataset, TaskRun, BBox, ComputerVisionSample
from ultralytics import YOLO

from src import detect_document, document_extractor, model, validation
from src.validation import SampleAccuracyResult, DatasetAccuracyResult
from src.image_segmentation import processMask, segmentDetections
from src.ocr import performOCR
from src.utils import savePlot, saveDocumentWithDetections
from src.object_detection import runObjectDetection


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

        sampleResult = validation.calculateAccuracy(sample, groundtruth, prediction, taskRun.parameters.get("iouThreshold"))

        logging.info(f"\tSample accuracy: {sampleResult.getAccuracy():.2f}")
        logging.info(f"\tAccuracy per label:")

        for labelResult in sampleResult.labelResults:
            logging.info(f"\t- {labelResult.name}: {labelResult.accuracy:.2f}")

        return sampleResult

    return None


def main() -> None:
    taskRun: TaskRun[ComputerVisionDataset] = currentTaskRun()
    taskRun.dataset.download()

    segmentationModel = model.loadSegmentationModel(taskRun.parameters["segmentationModel"])
    detectionModel = model.loadDetectionModel(taskRun.parameters["objectDetectionModel"])

    validationResult = DatasetAccuracyResult(taskRun.dataset.id, taskRun.dataset.name, [])

    for i, sample in enumerate(taskRun.dataset.samples):
        logging.info(f">> [Document OCR] Performing segmentation on sample \"{sample.name}\" ({i + 1}/{taskRun.dataset.count})")

        sampleResult = processSample(taskRun, sample, segmentationModel, detectionModel)
        if sampleResult is not None:
            validationResult.sampleResults.append(sampleResult)

    logging.info(f">> [Document OCR] Dataset accuracy: {validationResult.displayValue()}")

    # Create sample results artifact
    sampleResultsPath = folder_manager.temp / "results.csv"
    validationResult.writeSampleResults(sampleResultsPath)

    sampleResultsArtifact = taskRun.createArtifact(sampleResultsPath, "sample_results.csv")
    if sampleResultsArtifact is None:
        logging.error(">> [Document OCR] Failed to create Sample results artifact")

    # Create dataset result artifact
    datasetResultPath = folder_manager.temp / "dataset.csv"
    validationResult.writeDatasetResult(datasetResultPath)

    datasetResultArtifact = taskRun.createArtifact(datasetResultPath, "dataset_result.csv")
    if datasetResultArtifact is None:
        logging.error(">> [Document OCR] Failed to create Dataset result artifact")


if __name__ == "__main__":
    main()
