from typing import Any
from pathlib import Path

from coretex import functions

import detect_document
from model import loadSegmentationModel, getObjectDetectionModel
from image_segmentation import processMask, segmentImage, segmentDetections
from ocr import performOCR
from object_detection import runObjectDetection


DOCUMENT_NOT_FOUND = {
    "code": 500,
    "body": {
        "error": "Failed to find document on the provided image"
    }
}


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    modelsDir = requestData.get("model")

    segmentationModel = loadSegmentationModel(modelsDir / "segmentationModel")
    objDetModelWeights = getObjectDetectionModel(modelsDir / "objectDetectionModel")

    imagePath = requestData.get("image")
    if not isinstance(imagePath, Path):
        return functions.badRequest("Input image is invalid")

    predictedMask = detect_document.run(segmentationModel, imagePath)
    predictedMask = processMask(predictedMask)
    if predictedMask is None:
        return DOCUMENT_NOT_FOUND

    segmentedImage = segmentImage(imagePath, predictedMask)
    if segmentedImage is None:
        return DOCUMENT_NOT_FOUND

    bboxes, classes = runObjectDetection(segmentedImage, objDetModelWeights)
    segmentedDetections = segmentDetections(segmentedImage, bboxes)

    result = performOCR(segmentedDetections, classes)

    return functions.success({
        "result": result
    })
