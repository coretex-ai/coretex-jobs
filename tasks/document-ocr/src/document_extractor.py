from typing import Optional

from coretex import AnnotatedImageSampleData, ImageDatasetClasses, CoretexImageAnnotation, CoretexSegmentationInstance, BBox

import cv2
import numpy as np

from .point import Point2D
from .rect import Rect


def sortRectPoints(points: list[Point2D]) -> list[Point2D]:
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


def extractRectangle(mask: np.ndarray) -> Rect:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 1:
        raise ValueError("Found more than one contour")

    epsilon = 0.02 * cv2.arcLength(contours[0], True)
    rectangle = cv2.approxPolyDP(contours[0], epsilon, True)

    points: list[Point2D] = []
    for point in rectangle:
        points.append(Point2D(int(point[0][0]), int(point[0][1])))

    points = sortRectPoints(points)
    return Rect(*points)


def getDocumentInstance(annotation: CoretexImageAnnotation, classes: ImageDatasetClasses) -> CoretexSegmentationInstance:
    documentClass = classes.classByLabel("document")
    if documentClass is None:
        raise ValueError("Missing document class")

    for instance in annotation.instances:
        if instance.classId in documentClass.classIds:
            return instance

    raise ValueError("Failed to find document annotation")


def warpPerspectivePoint(matrix: np.ndarray, x: int, y: int) -> tuple[int, int]:
    px = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / ((matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]))
    py = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / ((matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]))

    return abs(px), abs(py)


def warpPerspectivePoly(transformMatrix: np.ndarray, points: list[int]) -> list[int]:
    result: list[int] = []

    for i in range(1, len(points), 2):
        x, y = warpPerspectivePoint(transformMatrix, points[i - 1], points[i])
        result.extend([x, y])

    return result


def extractDocumentImage(
    image: np.ndarray,
    mask: np.ndarray,
    classes: ImageDatasetClasses,
    annotation: Optional[CoretexImageAnnotation]
) -> tuple[np.ndarray, Optional[CoretexImageAnnotation]]:

    rect = extractRectangle(mask)

    documentSegmentation = [0, 0, rect.width, 0, rect.width, rect.height, 0, rect.height]
    transformed = np.array(documentSegmentation, dtype = np.float32).reshape((-1, 2))
    transformMatrix = cv2.getPerspectiveTransform(rect.numpy(), transformed)
    maskedImage = cv2.warpPerspective(image, transformMatrix, (rect.width, rect.height))

    transformedInstances: list[CoretexSegmentationInstance] = []

    if annotation is not None:
        documentClass = classes.classByLabel("document")
        if documentClass is None:
            raise ValueError("Missing document class")

        transformedInstances.append(CoretexSegmentationInstance.create(
            documentClass.classIds[0],
            BBox.fromPoly(documentSegmentation),
            [documentSegmentation]
        ))

        for instance in annotation.instances:
            if instance.classId in documentClass.classIds:
                continue

            transformedSegmentations = [
                warpPerspectivePoly(transformMatrix, segmentation)
                for segmentation in instance.segmentations
            ]
            flattenedSegmentations = [e for segmentation in transformedSegmentations for e in segmentation]

            transformedInstances.append(CoretexSegmentationInstance.create(
                instance.classId,
                BBox.fromPoly(flattenedSegmentations),
                transformedSegmentations
            ))

        transformedAnnotation = CoretexImageAnnotation.create(annotation.name, rect.width, rect.height, transformedInstances)
        return maskedImage, transformedAnnotation

    return maskedImage, None
