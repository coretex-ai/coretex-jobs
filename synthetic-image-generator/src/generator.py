from typing import Optional
import logging
import random

from PIL import Image
from PIL.Image import Image as PILImage
from numpy import ndarray

import numpy as np
import cv2

from coretex import (
    ImageDataset, ImageSample, AnnotatedImageSampleData, ImageDatasetClasses,
    CoretexImageAnnotation, CoretexSegmentationInstance
)

from .annotations import transformAnnotation
from .utils import Rect, uploadAugmentedImage


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
    segmentedImages: list[PILImage],
    backgroundImage: np.ndarray,
    angle: int,
    scale: float
) -> tuple[PILImage, list[tuple[int, int]], list[float]]:

    centroids: list[tuple[int, int]] = []
    locations: list[tuple[int, int, int, int]] = []
    scalingModifiers: list[float] = []

    background = Image.fromarray(backgroundImage)

    for segmentedImage in segmentedImages:
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

            if maxX <= 0 or maxY <= 0:
                logging.warning(">> [Image Stitching] Scaled image out of bounds. Reducing scaling by 20%")
                scalingModifier *= 0.8

                continue


            # Generate a random location within the bounds of the background image
            x = np.random.randint(0, maxX)
            y = np.random.randint(0, maxY)

            # Check if the image overlaps with any previously pasted images
            if isOverlapping(x, y, resizedImage, locations):
                scalingModifier *= 0.9
                continue

            break

        background.paste(resizedImage, (x, y), resizedImage)

        centerX = x + resizedImage.width // 2
        centerY = y + resizedImage.height // 2

        centroids.append((centerX, centerY))

        # Add the location to the list
        locations.append((x, y, resizedImage.width, resizedImage.height))

        scalingModifiers.append(scalingModifier)

    return background, centroids, scalingModifiers


def processInstance(
    sample: ImageSample,
    backgroundSampleData: AnnotatedImageSampleData,
    angle: int,
    scale: float,
    classes: ImageDatasetClasses,
    excludedClasses: list[str],
    unwarp: bool
) -> tuple[PILImage, list[CoretexSegmentationInstance]]:

    segmentedImages: list[Image.Image] = []
    transformMatrices: list[Optional[ndarray]] = []
    augmentedInstances: list[CoretexSegmentationInstance]= []

    sampleData = sample.load()

    annotation = sampleData.annotation
    if annotation is None:
        raise RuntimeError(f"CTX sample dataset sample id: {sample.id} image doesn't exist!")

    for instance in annotation.instances:
        if excludedClasses is not None and classes.classById(instance.classId).label in excludedClasses:
            continue

        foregroundMask = instance.extractBinaryMask(sampleData.image.shape[1], sampleData.image.shape[0])
        segmentedImage, transformMatrix = generateSegmentedImage(sampleData.image, foregroundMask, unwarp)
        segmentedImages.append(segmentedImage)
        transformMatrices.append(transformMatrix)

    composedImage, centroids, scalingModifiers = composeImage(segmentedImages, backgroundSampleData.image, angle, scale)

    # Proccess annotations
    index = 0
    for i, instance in enumerate(annotation.instances):
        if excludedClasses is not None and classes.classById(instance.classId).label in excludedClasses:
            continue

        binaryMask = instance.extractBinaryMask(annotation.width, annotation.height)

        augmentedInstances.append(transformAnnotation(
            instance,
            transformMatrix,
            scale * scalingModifiers[i],
            centroids[i],
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
            excludedClasses,
            unwarp
        )

        uploadAugmentedImage(f"{sample.id}-{i}", augmentedImage, annotations, outputDataset)
