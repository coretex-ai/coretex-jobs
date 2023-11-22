from typing import Optional
from pathlib import Path

import logging

from PIL import Image
from scipy import ndimage

import cv2
import numpy as np

from coretex import BBox


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

    return filledMask


def warpPerspective(image: np.ndarray, mask: np.ndarray) -> Optional[Image.Image]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 1:
            raise ValueError("Found more than one contour")

        epsilon = 0.02 * cv2.arcLength(contours[0], True)
        rectangle = cv2.approxPolyDP(contours[0], epsilon, True)
        if rectangle.shape[0] > 4:
            logging.error(">> [Document OCR] Failed to approximate rectangle. Mask may be too noisy")
            return None

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

        width = tr[0] - tl[0]
        height = bl[1] - tl[1]

        cornerPoints = np.array([tl, tr, br, bl], dtype = np.float32)

        transformed = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype = np.float32)
        transformMatrix = cv2.getPerspectiveTransform(cornerPoints, transformed)
        return Image.fromarray(cv2.warpPerspective(np.array(image, np.uint8), transformMatrix, (width, height)))


def segmentImage(
    image: Path,
    segmentationMask: np.ndarray
) -> Optional[Image.Image]:

    rgbaImage = Image.open(image).convert("RGBA")

    segmentedImage = np.asarray(rgbaImage) * segmentationMask[..., None]  # reshape segmentationMask for broadcasting

    segmentedImage = warpPerspective(segmentedImage, segmentationMask)
    if segmentedImage is None:
        return None

    alpha = segmentedImage.getchannel("A")
    bbox = alpha.getbbox()
    croppedImage = segmentedImage.crop(bbox)

    return croppedImage


def segmentDetections(
    image: Image.Image,
    bboxes: list[BBox]
) -> list[Image.Image]:

    segments: list[Image.Image] = []
    for i, bbox in enumerate(bboxes):
        segment = image.crop((bbox.minX, bbox.minY, bbox.maxX, bbox.maxY))
        segments.append(segment)

    return segments
