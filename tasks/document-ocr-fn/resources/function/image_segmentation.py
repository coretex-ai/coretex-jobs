from typing import Optional
from pathlib import Path

import math
import logging

from PIL import Image, ImageDraw
from scipy import ndimage

import cv2
import numpy as np

from coretex import BBox


def calculateDistance(centroid: tuple[int,int], point: tuple[int,int]) -> float:
    return math.sqrt(math.pow(centroid[0] - point[0], 2) + math.pow(centroid[1] - point[1], 2))


def reducePolygonTo4Points(points: list[tuple[int,int]]) -> list[tuple[int,int]]:
    xSum = sum([point[0] for point in points])
    ySum = sum([point[1] for point in points])

    centroid = (xSum / len(points), ySum / len(points))

    distances = [(point, calculateDistance(centroid, point)) for point in points]
    distances.sort(key = lambda x: x[1], reverse = True)

    # Return the four points furthest away from the centroid
    return [point for point, distance in distances[:4]]


def findRectangle(mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            logging.error("Failed to find document")
            return None

        contour = max(contours, key = cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        rectangle = cv2.approxPolyDP(contour, epsilon, True)

        points: list[tuple[int, int]] = []
        for point in rectangle:
            points.append((point[0][0], point[0][1]))

        if len(points) < 4:
            logging.error("Failed to find document")
            return None

        if len(points) > 4:
            points = reducePolygonTo4Points(points)

        xSorted = sorted(points, key = lambda point: point[0])

        left = xSorted[:2]
        left.sort(key = lambda point: point[1])
        tl, bl = left

        right = xSorted[2:]
        right.sort(key = lambda point: point[1])
        tr, br = right

        return np.array([tl, tr, br, bl], dtype = np.float32)


def processMask(predictedMask: np.ndarray) -> np.ndarray:
    # Erosion
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(predictedMask.astype(np.uint8), kernel, iterations=1)

    # Connected Component Analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(eroded_mask, connectivity=8)
    largestComponent = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    cleanMask = np.zeros_like(labels)
    cleanMask[labels == largestComponent] = 1

    # Hole Filling
    filledMask = ndimage.binary_fill_holes(cleanMask)

    # Finding quadrilateral
    cornerPoints = findRectangle(filledMask)
    if cornerPoints is None:
        return None

    width = predictedMask.shape[1]
    height = predictedMask.shape[0]

    polygon = [min(value, width - 1) if i % 2 == 0 else min(value, height - 1) for i, value in enumerate(cornerPoints.flatten())]
    polygon.extend((polygon[0], polygon[1]))  # Close polygon

    image = Image.new("L", (predictedMask.shape[1], predictedMask.shape[0]))

    draw = ImageDraw.Draw(image)
    draw.polygon(polygon, fill = 1)

    return np.array(image)


def segmentImage(
    image: Path,
    segmentationMask: np.ndarray
) -> Optional[Image.Image]:

    rgbaImage = Image.open(image).convert("RGBA")

    segmentedImage = (np.array(rgbaImage) * segmentationMask[..., None]).astype(np.uint8)  # reshape segmentationMask for broadcasting

    cornerPoints = findRectangle(segmentationMask)
    if cornerPoints is None:
        return None

    width = int(cornerPoints[1][0] - cornerPoints[0][0])  # Top left x minus top ritht x
    height = int(cornerPoints[3][1] - cornerPoints[0][1])  # Bottom left y minus top left y

    transformed = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = np.float32)
    transformMatrix = cv2.getPerspectiveTransform(cornerPoints, transformed)
    segmentedImage = Image.fromarray(cv2.warpPerspective(np.array(segmentedImage, np.uint8), transformMatrix, (width, height)))

    alpha = segmentedImage.getchannel("A")
    bbox = alpha.getbbox()
    croppedImage = segmentedImage.crop(bbox)

    return croppedImage


def segmentDetections(
    image: Image.Image,
    bboxes: list[BBox]
) -> list[Image.Image]:

    segments: list[Image.Image] = []
    for bbox in bboxes:
        segment = image.crop((bbox.minX, bbox.minY, bbox.maxX, bbox.maxY))
        segments.append(segment)

    return segments
