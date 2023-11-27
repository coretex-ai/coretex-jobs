from typing import Any
from pathlib import Path

from coretex import functions

import detect_document
from model import loadSegmentationModel, getObjectDetectionModel
from image_segmentation import processMask, segmentImage, segmentDetections
from ocr import performOCR
from object_detection import runObjectDetection


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    modelsDir = requestData.get("model")

    segmentationModel = loadSegmentationModel(modelsDir / "segmentationModel")
    objDetModelWeights = getObjectDetectionModel(modelsDir / "objectDetectionModel")

    imagePath = requestData.get("image")
    if not isinstance(imagePath, Path):
        return functions.badRequest("Input image is invalid")

    predictedMask = detect_document.run(segmentationModel, imagePath)
    predictedMask = processMask(predictedMask)

    segmentedImage = segmentImage(imagePath, predictedMask)
    if segmentedImage is None:
        return functions.badRequest("Failed to determine document borders")

    bboxes, classes = runObjectDetection(segmentedImage, objDetModelWeights)
    segmentedDetections = segmentDetections(segmentedImage, bboxes)

    result = performOCR(segmentedDetections, classes)

    return functions.success({
        "result": result
    })
