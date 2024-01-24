import math

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


def extractRectangle(mask: np.ndarray) -> Rect:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 1:
        raise ValueError(f"Found more than one contour ({len(contours)})")

    epsilon = 0.02 * cv2.arcLength(contours[0], True)
    approximatedPolygon = cv2.approxPolyDP(contours[0], epsilon, True)

    points: list[Point2D] = []
    for point in approximatedPolygon:
        points.append(Point2D(int(point[0][0]), int(point[0][1])))

    if len(points) < 4:
        raise ValueError(f"Approximated polygon to less than four points ({len(points)})")

    if len(points) > 4:
        points = reducePolygonTo4Points(points)

    points = sortRectPoints(points)

    return Rect(*points)
