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

from .utils import Rect, uploadAugmentedImage, getDistanceVector


SegmentationType = list[int]


def generateSegmentedImage(
    image: np.ndarray,
    segmentationMask: np.ndarray,
    unwarp: bool
) -> tuple[Image.Image, Optional[np.ndarray]]:

    rgbaImage = Image.fromarray(image).convert("RGBA")

    segmentedImage = np.asarray(rgbaImage) * segmentationMask[..., None]  # reshape segmentationMask for broadcasting
    segmentedImage = Image.fromarray(segmentedImage)

    if unwarp:
        rectangle = Rect.extractRectangle(segmentationMask)

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
    #centroid = Rect.extractRectangle(np.array(resizedImage)).center()

    return background, centroid, scalingModifier


def applyAffine(point: tuple[int, int], transformMatrix: ndarray) -> tuple[int, int]:
    columnVector = np.array([[point[0], point[1], 1]]).T
    hm = np.matmul(transformMatrix, columnVector).flatten()
    return (abs(int(hm[0] / hm[2])), abs(int(hm[1] / [hm[2]])))


def affineSegmentation(inputSegmentation: list[int], transformMatrix: np.ndarray) -> SegmentationType:
    outputSegmentation: list[int] = []
    for i in range(0, len(inputSegmentation), 2):
        cornerPoint = (inputSegmentation[i], inputSegmentation[i + 1])
        outputSegmentation.extend(applyAffine(cornerPoint, transformMatrix))

    return outputSegmentation


def transformAnnotation(
    instance: CoretexSegmentationInstance,
    transformMatrix: Optional[np.ndarray],
    scale: float,
    centroid: tuple[int, int],
    offsetVector: tuple[int, int],
    angle: int,
    unwarp: bool,
    mask: ndarray
) -> CoretexSegmentationInstance:

    segmentations = instance.segmentations
    if unwarp and transformMatrix is not None:
        points = Rect.extractRectangle(mask).numpy().flatten()
        segmentations = [np.append(points, [points[0], points[1]]).tolist()]
        segmentations = [affineSegmentation(sublist, transformMatrix) for sublist in segmentations]

    segmentationsScaled = [[value * scale for value in sublist] for sublist in segmentations]
    segmentationsFlattened = [sample for sublist in segmentationsScaled for sample in sublist]

    augmentedInstance = CoretexSegmentationInstance.create(
        instance.classId,
        BBox.fromPoly(segmentationsFlattened),
        segmentationsScaled
    )

    augmentedInstance.rotateSegmentations(angle)

    centroid = (centroid[0] - offsetVector[0] * scale, centroid[1] - offsetVector[1] * scale)
    augmentedInstance.centerSegmentations(centroid)

    return augmentedInstance


def processInstance(
    sample: ImageSample,
    backgroundSampleData: AnnotatedImageSampleData,
    angle: int,
    scale: float,
    classes: ImageDatasetClasses,
    documentClass: str,
    excludedClasses: list[str],
    unwarp: bool
) -> tuple[PILImage, list[CoretexSegmentationInstance]]:

    augmentedInstances: list[CoretexSegmentationInstance]= []

    sampleData = sample.load()

    annotation = sampleData.annotation
    if annotation is None:
        raise RuntimeError(f"CTX sample dataset sample id: {sample.id} image doesn't exist!")

    for instance in annotation.instances:
        if classes.classById(instance.classId).label != documentClass:
            continue

        foregroundMask = instance.extractBinaryMask(sampleData.image.shape[1], sampleData.image.shape[0])
        segmentedImage, transformMatrix = generateSegmentedImage(sampleData.image, foregroundMask, unwarp)

        documentCenter = Rect.extractRectangle(foregroundMask).center()

        composedImage, centroid, scalingModifier = composeImage(segmentedImage, backgroundSampleData.image, angle, scale)

    # Proccess annotations
    index = 0
    for instance in annotation.instances:
        if excludedClasses is not None and classes.classById(instance.classId).label in excludedClasses:
            continue

        binaryMask = instance.extractBinaryMask(annotation.width, annotation.height)
        center = Rect.extractRectangle(binaryMask).center()
        offsetVector = getDistanceVector(documentCenter, center, angle)
        offsetVector = applyAffine(offsetVector, transformMatrix)

        augmentedInstances.append(transformAnnotation(
            instance,
            transformMatrix,
            scale * scalingModifier,
            centroid,
            offsetVector,
            angle,
            unwarp,
            binaryMask
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
    excludedClasses: list[str],
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
        excludedClasses,
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
    augmentationsPerImage: int,
    angleLimit: int,
    scale: float,
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
            excludedClasses,
            unwarp
        )

        uploadAugmentedImage(f"{sample.id}-{i}", augmentedImage, annotations, outputDataset)
