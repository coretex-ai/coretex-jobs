import logging

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
