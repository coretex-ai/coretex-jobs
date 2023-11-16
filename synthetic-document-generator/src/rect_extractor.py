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
