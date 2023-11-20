import logging

from ultralytics import YOLO
from PIL.Image import Image

from coretex import BBox


def removeDuplicalteDetections(bboxes: list[BBox], classes: list[str]) -> tuple[list[BBox], list[str]]:
    newBboxes: list[BBox] = []
    newClasses: list[str] = []

    for i, bbox in enumerate(bboxes):
        if classes[i] in newClasses:
            logging.warning(f">> [Document OCR] Duplicate \"{classes[i]}\" detection will be discarded")
            continue

        newBboxes.append(bbox)
        newClasses.append(classes[i])

    return newBboxes, newClasses


def runObjectDetection(image: Image, model: YOLO) -> tuple[list[BBox], list[str]]:
    result = model(image)[0]

    bboxes = [BBox(
        int(xyxy[0]),
        int(xyxy[1]),
        int(xyxy[2] - xyxy[0]),
        int(xyxy[3] - xyxy[1])
    ) for xyxy in result.boxes.xyxy]

    classes = [result.names[int(clazz)] for clazz in result.boxes.cls]

    return removeDuplicalteDetections(bboxes, classes)
