from typing import Optional

import math

from PIL import Image
from coretex import ImageDatasetClasses, CoretexImageAnnotation, CoretexSegmentationInstance, BBox

import cv2
import numpy as np

from .rect import Rect
from .point import Point2D


def calculateDistance(centroid: Point2D, point: Point2D) -> float:
    return math.sqrt(math.pow(centroid.x - point.x, 2) + math.pow(centroid.y - point.y, 2))


def reducePolygonTo4Points(points: list[Point2D]) -> list[Point2D]:
    xSum = sum([point.x for point in points])
    ySum = sum([point.y for point in points])

    centroid = Point2D(xSum / len(points), ySum / len(points))

    distances = [(point, calculateDistance(centroid, point)) for point in points]
    distances.sort(key = lambda x: x[1], reverse = True)

    # Return the four points furthest away from the centroid
    return [point for point, distance in distances[:4]]


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

    if len(points) < 4:
        raise ValueError(f"Approximated polygon to less than four points ({len(points)})")

    if len(points) > 4:
        points = reducePolygonTo4Points(points)

    points = sortRectPoints(points)
    return Rect(*points)


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


def getParentInstance(annotation: CoretexImageAnnotation, classes: ImageDatasetClasses, parentClassName: str) -> CoretexSegmentationInstance:
    parentClass = classes.classByLabel(parentClassName)
    if parentClass is None:
        raise ValueError(f"Missing {parentClassName} class")

    for instance in annotation.instances:
        if instance.classId in parentClass.classIds:
            return instance

    raise ValueError(f"Failed to find {parentClassName} annotation")


def extractParent(
    image: np.ndarray,
    mask: np.ndarray,
    parentClassName: str,
    classes: ImageDatasetClasses,
    annotation: Optional[CoretexImageAnnotation]
) -> tuple[np.ndarray, Optional[CoretexImageAnnotation]]:

    rect = extractRectangle(mask)

    parentSegmentation = [0, 0, rect.width, 0, rect.width, rect.height, 0, rect.height]
    transformed = np.array(parentSegmentation, dtype = np.float32).reshape((-1, 2))
    transformMatrix = cv2.getPerspectiveTransform(rect.numpy(), transformed)
    parentImage = cv2.warpPerspective(image, transformMatrix, (rect.width, rect.height))

    transformedInstances: list[CoretexSegmentationInstance] = []

    if annotation is not None:
        parentClass = classes.classByLabel(parentClassName)
        if parentClass is None:
            raise ValueError(f"Missing {parentClassName} class")

        transformedInstances.append(CoretexSegmentationInstance.create(
            parentClass.classIds[0],
            BBox.fromPoly(parentSegmentation),
            [parentSegmentation]
        ))

        for instance in annotation.instances:
            if instance.classId in parentClass.classIds:
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
        return parentImage, transformedAnnotation

    return parentImage, None


def extractRegion(
    image: np.ndarray,
    segmentationMask: np.ndarray
) -> Image.Image:

    rgbaImage = Image.fromarray(image).convert("RGBA")

    segmentedImage = np.asarray(rgbaImage) * segmentationMask[..., None]  # reshape segmentationMask for broadcasting
    segmentedImage = Image.fromarray(segmentedImage)

    alpha = segmentedImage.getchannel("A")
    bbox = alpha.getbbox()
    croppedImage = segmentedImage.crop(bbox)

    return croppedImage
