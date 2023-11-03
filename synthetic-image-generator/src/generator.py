from typing import List, Tuple

import logging
import random

from PIL import Image
from PIL.Image import Image as PILImage
from numpy import ndarray

import numpy as np

from coretex import (
    ImageDataset, ImageSample, AnnotatedImageSampleData,
    CoretexImageAnnotation, CoretexSegmentationInstance, BBox
)

from .utils import uploadAugmentedImage


SegmentationType = List[int]


def isOverlapping(
    x: int,
    y: int,
    image: PILImage,
    locations: List[Tuple[int, int, int, int]]
) -> bool:

    for loc in locations:
        if (x < loc[0] + loc[2] and x + image.width > loc[0] and
            y < loc[1] + loc[3] and y + image.height > loc[1]):

            return True

    return False


def generateSegmentedImage(image: np.ndarray, segmentationMask: np.ndarray) -> Image:
    rgbaImage = Image.fromarray(image).convert("RGBA")

    segmentedImage = np.asarray(rgbaImage) * segmentationMask[..., None]  # reshape segmentationMask for broadcasting
    segmentedImage = Image.fromarray(segmentedImage)

    alpha = segmentedImage.getchannel("A")
    bbox = alpha.getbbox()
    croppedImage = segmentedImage.crop(bbox)

    return croppedImage


def composeImage(
    segmentedImages: List[PILImage],
    backgroundImage: np.ndarray,
    angle: int,
    scale: float
) -> Tuple[PILImage, List[Tuple[int, int]], List[float]]:

    centroids: List[Tuple[int, int]] = []
    locations: List[Tuple[int, int, int, int]] = []
    scalingModifiers: List[float] = []

    background = Image.fromarray(backgroundImage)

    for index, segmentedImage in enumerate(segmentedImages):
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

        while True:
            # Generate a random location within the bounds of the background image
            x = np.random.randint(0, maxX)
            y = np.random.randint(0, maxY)

            # Check if the image overlaps with any previously pasted images
            if not isOverlapping(x, y, resizedImage, locations):
                break

        background.paste(resizedImage, (x, y), resizedImage)

        centerX = x + resizedImage.width // 2
        centerY = y + resizedImage.height // 2

        centroids.append((centerX, centerY))

        # Add the location to the list
        locations.append((x, y, resizedImage.width, resizedImage.height))

        # Add the scaling modifier so that the anotation can be scaled accordingly
        scalingModifiers.append(scalingModifier)


    return background, centroids, scalingModifiers


def processInstance(
    sample: ImageSample,
    backgroundSampleData: AnnotatedImageSampleData,
    angle: int,
    scale: float
) -> Tuple[PILImage, List[CoretexSegmentationInstance]]:

    segmentedImages: List[PILImage] = []
    augmentedInstances: List[CoretexSegmentationInstance]= []

    sampleData = sample.load()
    if sampleData.annotation is None:
        raise RuntimeError(f"CTX sample dataset sample id: {sample.id} image doesn't exist!")

    for instance in sampleData.annotation.instances:
        foregroundMask = instance.extractBinaryMask(sampleData.image.shape[1], sampleData.image.shape[0])
        segmentedImage = generateSegmentedImage(sampleData.image, foregroundMask)

        segmentedImages.append(segmentedImage)

    composedImage, centroids, scalingModifiers = composeImage(segmentedImages, backgroundSampleData.image, angle, scale)

    for i, instance in enumerate(sampleData.annotation.instances):
        anotationSegmentations = [[sample * scale * scalingModifiers[i] for sample in sublist] for sublist in instance.segmentations]
        segmentationsFlatten = [sample for sublist in anotationSegmentations for sample in sublist]

        augmentedInstance = CoretexSegmentationInstance.create(instance.classId, BBox.fromPoly(segmentationsFlatten), anotationSegmentations)

        augmentedInstance.rotateSegmentations(angle)
        augmentedInstance.centerSegmentations(centroids[i])

        augmentedInstances.append(augmentedInstance)

    return composedImage, augmentedInstances


def processSample(
    sample: ImageSample,
    backgroundSample: ImageSample,
    angle: int,
    scale: float
) -> Tuple[ndarray, CoretexImageAnnotation]:

    backgroundSampleData = backgroundSample.load()

    composedImage, augmentedInstances = processInstance(sample, backgroundSampleData, angle, scale)
    annotation = CoretexImageAnnotation.create(sample.name, composedImage.width, composedImage.height, augmentedInstances)

    return composedImage, annotation


def augmentSample(
    sample: ImageDataset,
    backgroundDataset: ImageDataset,
    angleLimit: int,
    scale: float,
    augmentationsPerImage: int,
    outputDataset: ImageDataset
) -> None:

    for i in range(augmentationsPerImage):
        logging.info(f">> [Image Stitching] Stitching image {i}")
        background = backgroundDataset.samples[random.randint(0, backgroundDataset.count - 1)]
        background.unzip()

        angle = random.randint(-angleLimit, angleLimit)

        augmentedImage, annotations = processSample(sample, background, angle, scale)
        uploadAugmentedImage(f"{sample.id}-{i}", augmentedImage, annotations, outputDataset)
