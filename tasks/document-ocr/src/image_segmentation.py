from typing import Optional
from pathlib import Path

import logging

from PIL import Image, ImageDraw
from scipy import ndimage

import cv2
import numpy as np

from coretex import BBox, TaskRun

from .utils import createArtifact


def findRectangle(mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            logging.error("Failed to find predicted mask")
            return

        contour = max(contours, key = cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        rectangle = cv2.approxPolyDP(contour, epsilon, True)

        if rectangle.shape[0] > 4:
            rectangle = cv2.minAreaRect(contour)
            return cv2.boxPoints(rectangle)

        points: list[list[int]] = []
        for point in rectangle:
            points.append([point[0][0], point[0][1]])

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
    bboxes: list[BBox],
    classes: list[str],
    outputDir: Path,
    taskRun: TaskRun
) -> list[Image.Image]:

    segments: list[Image.Image] = []
    for i, bbox in enumerate(bboxes):
        segment = image.crop((bbox.minX, bbox.minY, bbox.maxX, bbox.maxY))

        classPath = outputDir / classes[i]
        classSegmentPath = classPath / "image.png"
        classPath.mkdir(parents = True, exist_ok = True)

        segment.save(classSegmentPath)
        createArtifact(taskRun, classSegmentPath, classSegmentPath.relative_to(outputDir.parent))

        segments.append(segment)

    return segments
