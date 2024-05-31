from typing import Optional, Any

import logging

import numpy as np

import imageio.v3 as imageio

from coretex import CoretexImageAnnotation, ImageDataset, folder_manager, ImageSample, TaskRun


def uploadAugmentedImage(
    imageName: str,
    augmentedImage: np.ndarray,
    annotation: CoretexImageAnnotation,
    metadata: dict[str, Any],
    taskRun: TaskRun,
    outputDataset: ImageDataset
) -> None:

    imagePath = folder_manager.temp / imageName
    imageio.imwrite(imagePath, augmentedImage)

    try:
        augmentedSample = outputDataset.add(imagePath)
    except BaseException as ex:
        logging.error(f">> [Image Augmentation] Failed to upload sample {imagePath} - \"{ex}\"")
        return

    if not augmentedSample.saveAnnotation(annotation):
        logging.error(f">> [Image Augmentation] Failed to update sample annotation {imagePath}")

    augmentedSample.saveMetadata(metadata)
    taskRun.createArtifact(imagePath, imagePath.name)


def copySample(sample: ImageSample, dataset: ImageDataset) -> None:
    sample.unzip()

    try:
        copy = dataset.add(sample.imagePath, sample.name)
    except BaseException as ex:
        logging.error(f"\tFailed to copy sample \"{sample.name}\" - \"{ex}\"")
        return

    annotation = sample.load().annotation
    if annotation is not None:
        if not copy.saveAnnotation(annotation):
            logging.error("\tFailed to copy sample annotation, deleting...")

            if not copy.delete():
                logging.error("\tFailed to delete sample")


def getRelevantParameters(taskRun: TaskRun) -> list[str]:
    relevantParams = taskRun.parameters.copy()

    relevantParams["dataset"] = relevantParams["dataset"].id
    relevantParams.pop("outputDataset")
    return [str(value) for value in relevantParams.values()]


def getCache(cacheName: str, expectedSize: int) -> Optional[ImageDataset]:
    caches = ImageDataset.fetchAll(name = cacheName, include_sessions = 1)
    for cache in caches:
        if cache.count == expectedSize:
            logging.info(">> [Image Augmentation] Cache found!")
            return cache

    return None


def convertToSerializable(obj):
    if isinstance(obj, dict):
        return {k: convertToSerializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertToSerializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convertToSerializable(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
