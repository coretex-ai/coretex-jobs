from ultralytics import YOLO
from PIL.Image import Image

import numpy as np

from coretex import BBox


def removeDuplicalteDetections(
    bboxes: list[BBox],
    classes: list[str],
    confidence: list[float]
) -> tuple[list[BBox], list[str]]:

    newBboxes: list[BBox] = []
    newClasses = np.unique(classes).tolist()

    for uniqueClass in newClasses:
        classConfs = np.where(np.array(classes) == uniqueClass, confidence, 0)
        newBboxes.append(bboxes[classConfs.argmax()])

    return newBboxes, newClasses


def runObjectDetection(image: Image, model: YOLO) -> tuple[list[BBox], list[str]]:
    result = model(image)[0]

    bboxes = [BBox.create(
        int(xyxy[0]),
        int(xyxy[1]),
        int(xyxy[2]),
        int(xyxy[3])
    ) for xyxy in result.boxes.xyxy]

    classes = [result.names[int(clazz)] for clazz in result.boxes.cls]

    confidence = list(result.boxes.conf)

    return removeDuplicalteDetections(bboxes, classes, confidence)
