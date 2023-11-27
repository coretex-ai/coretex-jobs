from typing import Optional

import logging
import math

from PIL.Image import Image as PILImage

import cv2
import numpy as np

from coretex import ImageSample, ImageDataset, CoretexImageAnnotation, folder_manager


ANNOTATION_NAME = "annotations.json"


class Point2D:

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class Rect:

    def __init__(self, tl: Point2D, tr: Point2D, br: Point2D, bl: Point2D) -> None:
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl

    @property
    def witdh(self) -> int:
        return self.tr.x - self.tl.x

    @property
    def height(self) -> int:
        return self.bl.y - self.tl.y

    def numpy(self) -> np.ndarray:
        return np.array([
            [self.tl.x, self.tl.y],
            [self.tr.x, self.tr.y],
            [self.br.x, self.br.y],
            [self.bl.x, self.bl.y]
        ], dtype = np.float32)

    def center(self) -> tuple[int, int]:
        temp = np.sum(self.numpy(), axis = 0) // 4
        return temp

    @classmethod
    def sortRectPoints(cls, points: list[Point2D]) -> list[Point2D]:
        xSorted = sorted(points, key = lambda point: point.x)

        left = xSorted[:2]
        left.sort(key = lambda point: point.y)
        tl, bl = left

        right = xSorted[2:]
        right.sort(key = lambda point: point.y)
        tr, br = right

        return [
            tl, tr, br, bl
        ]

    @classmethod
    def extractRectangle(cls, mask: np.ndarray) -> 'Rect':
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 1:
            raise ValueError("Found more than one contour")

        epsilon = 0.02 * cv2.arcLength(contours[0], True)
        rectangle = cv2.approxPolyDP(contours[0], epsilon, True)

        points: list[Point2D] = []
        for point in rectangle:
            points.append(Point2D(point[0][0], point[0][1]))

        points = cls.sortRectPoints(points)
        return cls(*points)


def uploadAugmentedImage(
    imageName: str,
    augmentedImage: PILImage,
    annotation: CoretexImageAnnotation,
    outputDataset: ImageDataset
) -> None:

    imagePath = folder_manager.temp / f"{imageName}.jpeg"
    augmentedImage.save(imagePath)

    augmentedSample = ImageSample.createImageSample(outputDataset.id, imagePath)
    if augmentedSample is None:
        logging.error(f">> [Image Stitching] Failed to upload sample {imagePath}")
        return

    augmentedSample.download()
    augmentedSample.unzip()
    if not augmentedSample.saveAnnotation(annotation):
        logging.error(f">> [Image Stitching] Failed to update sample annotation {imagePath}")

    imagePath.unlink(missing_ok = True)
