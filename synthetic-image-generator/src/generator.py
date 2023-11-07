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


def getCornerPoints(segmentation: list[int]):
    return [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2) if i < 8]

    simplifedPoints = approximate_polygon(points, tolerance = 0.1)
    subdividedPoints = subdivide_polygon(simplifedPoints)

    # # PCA
    # center = np.mean(subdividedPoints, axis = 0)
    # subdividedPointsCentered = subdividedPoints - center
    # covarianceMatrix = np.dot(subdividedPointsCentered.T, subdividedPointsCentered) / len(subdividedPointsCentered)
    # eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
    # sortedIndices = np.argsort(eigenvalues)[::-1]
    # eigenvectorsSorted = eigenvectors[:, sortedIndices]
    # xVector = eigenvectorsSorted[:, 0]
    # yVector = eigenvectorsSorted[:, 1]

    # cornerPoints = [
    #     center + xVector * np.min(subdividedPointsCentered.dot(xVector)),
    #     center + xVector * np.max(subdividedPointsCentered.dot(xVector)),
    #     center + yVector * np.min(subdividedPointsCentered.dot(yVector)),
    #     center + yVector * np.max(subdividedPointsCentered.dot(yVector))
    # ]

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype = np.int32)

    # Extract the four corner points
    cornerPoints = box.tolist()

    return cornerPoints


def generateSegmentedImage(
    image: np.ndarray,
    segmentationMask: np.ndarray,
    segmentation: list[int],
    unwarp: bool
) -> tuple[Image.Image, Optional[np.ndarray]]:

    rgbaImage = Image.fromarray(image).convert("RGBA")

    segmentedImage = np.asarray(rgbaImage) * segmentationMask[..., None]  # reshape segmentationMask for broadcasting
    segmentedImage = Image.fromarray(segmentedImage)

    if unwarp:
        cornerPoints = getCornerPoints(segmentation)

        width = int(np.average([math.dist(cornerPoints[0], cornerPoints[1]), math.dist(cornerPoints[2], cornerPoints[3])]))
        height = int(np.average([math.dist(cornerPoints[0], cornerPoints[3]), math.dist(cornerPoints[1], cornerPoints[2])]))

        rectPoints = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        transformMatrix = cv2.getPerspectiveTransform(np.array(cornerPoints, dtype = np.float32), rectPoints)
        segmentedImage = Image.fromarray(cv2.warpPerspective(np.array(segmentedImage), transformMatrix, (width, height)))
    else:
        transformMatrix = None

    alpha = segmentedImage.getchannel("A")
    bbox = alpha.getbbox()
    croppedImage = segmentedImage.crop(bbox)

    return croppedImage, transformMatrix


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
    unwarp: bool
) -> CoretexSegmentationInstance:

    segmentations = instance.segmentations
    if unwarp:
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
    unwarp: bool
) -> tuple[PILImage, list[CoretexSegmentationInstance]]:

    augmentedInstances: list[CoretexSegmentationInstance]= []

    sampleData = sample.load()
    if sampleData.annotation is None:
        raise RuntimeError(f"CTX sample dataset sample id: {sample.id} image doesn't exist!")

    for instance in sampleData.annotation.instances:
        if classes.classById(instance.classId).label != documentClass:
            continue

        oldDocCentroid = instance.centroid()

        foregroundMask = instance.extractBinaryMask(sampleData.image.shape[1], sampleData.image.shape[0])
        segmentedImage, transformMatrix = generateSegmentedImage(
            sampleData.image,
            foregroundMask,
            instance.segmentations[0],
            unwarp
        )

        composedImage, centroid, scalingModifier = composeImage(segmentedImage, backgroundSampleData.image, angle, scale)

    # Proccess info annotations
    index = 0
    for instance in sampleData.annotation.instances:
        offsetVector = getDistanceVector(oldDocCentroid, instance.centroid())
        offsetVector = rotateVector(offsetVector, angle)

        augmentedInstances.append(transformAnnotation(
            instance,
            transformMatrix,
            scale * scalingModifier,
            centroid,
            offsetVector,
            angle,
            unwarp
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
    unwarp: bool
) -> tuple[ndarray, CoretexImageAnnotation]:

    backgroundSampleData = backgroundSample.load()

    composedImage, augmentedInstances = processInstance(
        sample,
        backgroundSampleData,
        angle,
        scale,
        classes,
        documentClass,
        unwarp
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
            unwarp
        )

        uploadAugmentedImage(f"{sample.id}-{i}", augmentedImage, annotations, outputDataset)
