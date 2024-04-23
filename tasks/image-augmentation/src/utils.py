from typing import Optional

import logging

from numpy import ndarray

import imageio.v3 as imageio

from coretex import CoretexImageAnnotation, ImageDataset, folder_manager, ImageSample, TaskRun
from coretex.utils import hashCacheName


def uploadAugmentedImage(
    imageName: str,
    augmentedImage: ndarray,
    annotation: CoretexImageAnnotation,
    metadata: dict,
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
        copy = dataset.add(sample.imagePath)
    except BaseException as ex:
        logging.error(f"\tFailed to copy sample \"{sample.name}\" - \"{ex}\"")
        return

    annotation = sample.load().annotation
    if annotation is not None:
        if not copy.saveAnnotation(annotation):
            logging.error("\tFailed to copy sample annotation, deleting...")

            if not copy.delete():
                logging.error("\tFailed to delete sample")


def getOutputDatasetName(taskRun: TaskRun) -> str:
    relevantParams = taskRun.parameters.copy()

    relevantParams["dataset"] = relevantParams["dataset"].id
    relevantParams.pop("outputDataset")

    return hashCacheName(f"{taskRun.id}-AugImg", ".".join(str(relevantParams.values())))


def getCache(cacheName: str, expectedSize: int) -> Optional[ImageDataset]:
    caches = ImageDataset.fetchAll(name = cacheName, include_sessions = 1)
    for cache in caches:
        if cache.count == expectedSize:
            logging.info(">> [Image Augmentation] Cache found!")
            return cache

    return None
