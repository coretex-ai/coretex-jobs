from typing import Any
from pathlib import Path

from coretex import functions

import detect_document
from model import loadSegmentationModel, getWeights
from image_segmentation import processMask, segmentImage, segmentDetections
from ocr import performOCR
from utils import removeDuplicalteDetections
from object_detection.detect import run as runObjectDetection


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    modelsDir = requestData.get("model")
    segmentationModel = loadSegmentationModel(modelsDir / "segmentationModel")
    objDetModelWeights = getWeights(modelsDir / "objectDetectionModel")

    imagePath = requestData.get("image")
    if not isinstance(imagePath, Path):
        return functions.badRequest("Input image is invalid")

    predictedMask = detect_document.run(segmentationModel, imagePath)
    predictedMask = processMask(predictedMask)

    segmentedImage = segmentImage(imagePath, predictedMask)
    if segmentedImage is None:
        return functions.badRequest("Failed to determine document borders")


    bboxes, classes = runObjectDetection(segmentedImage, objDetModelWeights)
    bboxes, classes = removeDuplicalteDetections(bboxes, classes)

    segmentedDetections, labels = segmentDetections(segmentedImage, bboxes, classes)

    result = performOCR(segmentedDetections, labels)

    return functions.success({
        "result": result
    })
