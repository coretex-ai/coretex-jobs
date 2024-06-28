from typing import Optional

import logging

import cv2
import numpy as np
import imgaug.augmenters as iaa
import imageio.v3 as imageio

from coretex import ImageDataset, ImageSample, CoretexSegmentationInstance, BBox, CoretexImageAnnotation, AnnotatedImageSampleData

from .utils import uploadAugmentedImage


def mask2poly(mask: np.ndarray) -> Optional[list[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        logging.warning(">> [Image Augmentation] Could not find annotated area on augmented image")
        return None

    epsilon = 0.02 * cv2.arcLength(contours[0], True)
    poly = cv2.approxPolyDP(contours[0], epsilon, True)

    segmentation: list[int] = []
    for point in poly:
        segmentation.append(int(point[0][0]))
        segmentation.append(int(point[0][1]))

    return segmentation


def transformAnnotationInstances(sampleData: AnnotatedImageSampleData, pipeline: iaa.Sequential) -> Optional[list[CoretexSegmentationInstance]]:
    augmentedInstances: list[CoretexSegmentationInstance] = []

    annotation = sampleData.annotation
    if annotation is None:
        return None

    for instance in annotation.instances:
        mask = instance.extractSegmentationMask(
            annotation.width,
            annotation.height
        )

        mask = np.repeat(mask[..., None] * 255, 3, axis = -1)

        augmentedMask = pipeline.augment_image(mask)
        augmentedMask = (np.average(augmentedMask, axis = -1) > 127).astype(int)

        newSegmentations = mask2poly(augmentedMask)
        if newSegmentations is None:
            continue

        augmentedInstances.append(CoretexSegmentationInstance.create(
            instance.classId,
            BBox.fromPoly(newSegmentations),
            [newSegmentations]
        ))

    return augmentedInstances


def augmentImage(
    firstPipeline: iaa.Sequential,
    secondPipeline: iaa.Sequential,
    sample: ImageSample,
    numOfImages: int,
    outputDataset: ImageDataset
) -> None:

    sample.unzip()

    image = imageio.imread(sample.imagePath)
    sampleData = sample.load()

    for i in range(numOfImages):
        firstPipeline_ = firstPipeline.localize_random_state()
        firstPipeline_ = firstPipeline_.to_deterministic()

        augmentedImage = firstPipeline_.augment_image(image)
        augmentedImage = secondPipeline.augment_image(augmentedImage)
        augmentedInstances = transformAnnotationInstances(sampleData, firstPipeline_)
        if augmentedInstances is not None:
            annotation = CoretexImageAnnotation.create(
                sample.name,
                augmentedImage.shape[1],
                augmentedImage.shape[0],
                augmentedInstances
            )
        else:
            annotation = None

        augmentedImageName = f"{sample.name}-{i}" + sample.imagePath.suffix
        uploadAugmentedImage(augmentedImageName, augmentedImage, annotation, outputDataset)
