from typing import Optional
import logging
import random
import math

from PIL import Image
from PIL.Image import Image as PILImage
from numpy import ndarray

import numpy as np
import cv2

from coretex import (
    ImageDataset, ImageSample, AnnotatedImageSampleData, ImageDatasetClasses,
    CoretexImageAnnotation, CoretexSegmentationInstance, BBox
)

from .utils import uploadAugmentedImage


SegmentationType = list[int]


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
        points.append(Point2D(point[0][0], point[0][1]))

    points = sortRectPoints(points)
    return Rect(*points)


def generateSegmentedImage(
    image: np.ndarray,
    segmentationMask: np.ndarray,
    unwarp: bool
) -> tuple[Image.Image, Optional[np.ndarray]]:

    rgbaImage = Image.fromarray(image).convert("RGBA")

    segmentedImage = np.asarray(rgbaImage) * segmentationMask[..., None]  # reshape segmentationMask for broadcasting
    segmentedImage = Image.fromarray(segmentedImage)

    if unwarp:
        rectangle = extractRectangle(segmentationMask)

        width = rectangle.witdh
        height = rectangle.height

        transformed = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        transformMatrix = cv2.getPerspectiveTransform(rectangle.numpy(), transformed)
        segmentedImage = Image.fromarray(cv2.warpPerspective(np.array(segmentedImage), transformMatrix, (width, height)))
    else:
        transformMatrix = None

    alpha = segmentedImage.getchannel("A")
    bbox = alpha.getbbox()
    croppedImage = segmentedImage.crop(bbox)

    return croppedImage, transformMatrix


def isOverlapping(
    x: int,
    y: int,
    image: PILImage,
    locations: list[tuple[int, int, int, int]]
) -> bool:

    for loc in locations:
        if (x < loc[0] + loc[2] and x + image.width > loc[0] and
            y < loc[1] + loc[3] and y + image.height > loc[1]):

            return True

    return False


def composeImage(
    segmentedImage: PILImage,
    backgroundImage: np.ndarray,
    angle: int,
    scale: float
) -> tuple[PILImage, list[tuple[int, int]], list[float]]:

    background = Image.fromarray(backgroundImage)

    image = segmentedImage
    scalingModifier = 1.0

    while True:
        rotatedImage = image.rotate(angle, expand = True)
        resizedImage = rotatedImage.resize((
            int(rotatedImage.width * scale * scalingModifier),
            int(rotatedImage.height * scale * scalingModifier)
        ))

        # Calculate the maximum x and y coordinates for the top left corner of the image
        maxX = background.width - resizedImage.width
        maxY = background.height - resizedImage.height

        if maxX > 0 and maxY > 0:
            break

        logging.warning(">> [Image Stitching] Scaled image out of bounds. Reducing scaling by 20%")
        scalingModifier *= 0.8

    # Generate a random location within the bounds of the background image
    x = np.random.randint(0, maxX)
    y = np.random.randint(0, maxY)

    background.paste(resizedImage, (x, y), resizedImage)

    centerX = x + resizedImage.width // 2
    centerY = y + resizedImage.height // 2

    centroid = (centerX, centerY)

    return background, centroid, scalingModifier


def applyAffine(inputSegmentation: list[int], transformMatrix: np.ndarray) -> SegmentationType:
    outputSegmentation: list[int] = []
    for i in range(0, len(inputSegmentation), 2):
        cornerPoint = np.array([[inputSegmentation[i], inputSegmentation[i + 1], 1]]).T
        hm = np.matmul(transformMatrix, cornerPoint).flatten()

        outputSegmentation.extend((abs(int(hm[0] / hm[2])), abs(int(hm[1] / [hm[2]]))))

    return outputSegmentation


def transformAnnotation(
    instance: CoretexSegmentationInstance,
    transformMatrix: np.ndarray,
    scale: float,
    centroid: tuple[int, int],
    offsetVector: tuple[int, int],
    angle: int,
    unwarp: bool,
    mask: ndarray
) -> CoretexSegmentationInstance:

    segmentations = instance.segmentations
    if unwarp:
        rect = extractRectangle(mask).numpy().flatten()
        segmentations = [list(np.append(rect, [rect[0], rect[1]]))]
        segmentations = [applyAffine(sublist, transformMatrix) for sublist in segmentations]

    segmentationsScaled = [[value * scale for value in sublist] for sublist in segmentations]
    segmentationsFlattened = [sample for sublist in segmentationsScaled for sample in sublist]

    augmentedInstance = CoretexSegmentationInstance.create(
        instance.classId,
        BBox.fromPoly(segmentationsFlattened),
        segmentationsScaled
    )

    augmentedInstance.rotateSegmentations(angle)
    centroid = (centroid[0] + offsetVector[0], centroid[1] + offsetVector[1])  # the vector doesn't account for affine
    augmentedInstance.centerSegmentations(centroid)  #@# << 0 -> i

    return augmentedInstance


def getDistanceVector(point1: tuple[int, int], point2: tuple[int, int]) -> tuple[int, int]:
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]

    return (x, y)


def rotateVector(vector: tuple[int, int], degrees: int) -> tuple[int, int]:
    x, y = vector

    theta = math.radians(-degrees)
    cosang, sinang = math.cos(theta), math.sin(theta)

    newX = int(x * cosang - y * sinang)
    newY = int(x * sinang + y * cosang)

    return (newX, newY)


def processInstance(
    sample: ImageSample,
    backgroundSampleData: AnnotatedImageSampleData,
    angle: int,
    scale: float,
    classes: ImageDatasetClasses,
    documentClass: str,
    unwarp: bool,
    excludedClasses: list[str]
) -> tuple[PILImage, list[CoretexSegmentationInstance]]:

    augmentedInstances: list[CoretexSegmentationInstance]= []

    sampleData = sample.load()

    annotation = sampleData.annotation
    if annotation is None:
        raise RuntimeError(f"CTX sample dataset sample id: {sample.id} image doesn't exist!")

    for instance in annotation.instances:
        if classes.classById(instance.classId).label != documentClass:
            continue

        oldDocCentroid = instance.centroid()

        foregroundMask = instance.extractBinaryMask(sampleData.image.shape[1], sampleData.image.shape[0])
        segmentedImage, transformMatrix = generateSegmentedImage(sampleData.image, foregroundMask, unwarp)

        composedImage, centroid, scalingModifier = composeImage(segmentedImage, backgroundSampleData.image, angle, scale)

    # Proccess info annotations
    index = 0
    for instance in annotation.instances:
        if excludedClasses is not None and classes.classById(instance.classId).label in excludedClasses:
            continue

        offsetVector = getDistanceVector(oldDocCentroid, instance.centroid())
        offsetVector = rotateVector(offsetVector, angle)

        augmentedInstances.append(transformAnnotation(
            instance,
            transformMatrix,
            scale * scalingModifier,
            centroid,
            offsetVector,
            angle,
            unwarp,
            instance.extractBinaryMask(annotation.width, annotation.height)
        ))

        index += 1

    return composedImage, augmentedInstances


def processSample(
    sample: ImageSample,
    backgroundSample: ImageSample,
    angle: int,
    scale: float,
    classes: ImageDatasetClasses,
    documentClass: str,
    unwarp: bool,
    excludedClasses: list[str]
) -> tuple[ndarray, CoretexImageAnnotation]:

    backgroundSampleData = backgroundSample.load()

    composedImage, augmentedInstances = processInstance(
        sample,
        backgroundSampleData,
        angle,
        scale,
        classes,
        documentClass,
        unwarp,
        excludedClasses
    )
    annotation = CoretexImageAnnotation.create(
        sample.name,
        composedImage.width,
        composedImage.height,
        augmentedInstances
    )

    return composedImage, annotation


def augmentSample(
    sample: ImageDataset,
    backgroundDataset: ImageDataset,
    angleLimit: int,
    scale: float,
    augmentationsPerImage: int,
    classes: ImageDatasetClasses,
    documentClass: str,
    unwarp: bool,
    excludedClasses: list[str],
    outputDataset: ImageDataset
) -> None:

    for i in range(augmentationsPerImage):
        logging.info(f">> [Image Stitching] Stitching image {i}")
        background = backgroundDataset.samples[random.randint(0, backgroundDataset.count - 1)]
        background.unzip()

        angle = random.randint(-angleLimit, angleLimit)

        augmentedImage, annotations = processSample(
            sample,
            background,
            angle,
            scale,
            classes,
            documentClass,
            unwarp,
            excludedClasses
        )

        uploadAugmentedImage(f"{sample.id}-{i}", augmentedImage, annotations, outputDataset)
