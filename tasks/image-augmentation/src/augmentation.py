from typing import Any

import logging

import cv2
import numpy as np
import albumentations as A

from coretex import ImageDataset, ImageSample, CoretexSegmentationInstance, BBox, CoretexImageAnnotation, TaskRun

from .utils import uploadAugmentedImage


def mask2poly(mask: np.ndarray) -> list[int]:
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


def performAugmentation(
    image: np.ndarray,
    transform: A.ReplayCompose,
    annotation: CoretexImageAnnotation
) -> tuple[np.ndarray, CoretexImageAnnotation, dict[str: Any]]:

    augmentedInstances: list[CoretexSegmentationInstance] = []

    data = {"image": image}
    for i, instance in enumerate(annotation.instances):
        mask = instance.extractSegmentationMask(
            annotation.width,
            annotation.height
        )
        mask = np.repeat(mask[..., None] * 255, 3, axis = -1)
        data[f"mask{i}"] = mask

    transformed = transform(**data)
    augmentedImage = transformed["image"]
    metadata = transformed["replay"]
    augmentedMasks = [transformed[f"mask{i}"] for i in range(len(annotation.instances))]

    for instance, augmentedMask in zip(annotation.instances, augmentedMasks):
        augmentedMask = (np.average(augmentedMask, axis = -1) > 127).astype(int)

        newSegmentations = mask2poly(augmentedMask)
        if newSegmentations is None:
            logging.error(f"[Image Augmentation] Failed to transfer annotation. {annotation.name}")
            continue

        augmentedInstances.append(CoretexSegmentationInstance.create(
            instance.classId,
            BBox.fromPoly(newSegmentations),
            [newSegmentations]
        ))

    return augmentedImage, augmentedInstances, metadata


def processMetadata(inputMetadata: dict[str, Any]) -> dict[str, Any]:
    for transform in inputMetadata["transforms"]:
        if transform["__class_fullname__"] == "GaussNoise" and transform["applied"]:
            transform["params"]["gauss"] = None  # Hits timeout during upload

    outputMetadata = {"full_augmentation_metadata": inputMetadata}  # Include the full original metadata
    transforms = inputMetadata.get("transforms", [])

    for transform in transforms:
        className = transform.get("__class_fullname__", "")
        appliedStatus = transform.get("applied", False)
        outputMetadata[className] = appliedStatus

    return outputMetadata


def augmentImage(
    augmentersGeometric: list[A.BaseCompose],
    transformPhotometric: A.ReplayCompose,
    sample: ImageSample,
    taskRun: TaskRun,
    outputDataset: ImageDataset
) -> None:

    sample.unzip()

    sampleData = sample.load()
    image = sampleData.image
    annotation = sampleData.annotation

    for i in range(taskRun.parameters["numOfImages"]):
        # Dynamicaly build the Compose, so it can transform any number of annotation
        additionalTargets = {f"mask{i}": "image" for i in range(len(annotation.instances))}
        transformGeometric = A.ReplayCompose(augmentersGeometric, additional_targets = additionalTargets)
        augmentedImage, augmentedInstances, metadata = performAugmentation(image, transformGeometric, annotation)

        transformedPhotometric = transformPhotometric(image = augmentedImage)
        augmentedImage = transformedPhotometric["image"]

        metadata["transforms"] = metadata["transforms"] + transformedPhotometric["replay"]["transforms"]
        metadata = processMetadata(metadata)

        newAnnotation = CoretexImageAnnotation.create(
            sample.name,
            augmentedImage.shape[1],
            augmentedImage.shape[0],
            augmentedInstances
        )

        augmentedImageName = f"{sample.name}-{i}" + sample.imagePath.suffix
        uploadAugmentedImage(augmentedImageName, augmentedImage, newAnnotation, metadata, taskRun, outputDataset)
