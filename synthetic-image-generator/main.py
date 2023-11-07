from typing import Optional

import logging

from coretex import currentTaskRun, ImageDataset, TaskRun
from coretex.utils import hashCacheName

from src.generator import augmentSample


def getOutputDatasetName(taskRun: TaskRun):
    relevantParams: list[str] = []

    relevantParams.append(str(taskRun.parameters["dataset"].id))
    relevantParams.append(str(taskRun.parameters["backgroundDataset"].id))
    relevantParams.append(str(taskRun.parameters["augmentationsPerImage"]))
    relevantParams.append(str(taskRun.parameters["rotation"]))
    relevantParams.append(str(taskRun.parameters["scaling"]))

    return hashCacheName(f"{taskRun.id}-SynthImg", ".".join(relevantParams))


def getCache(cacheName: str) -> Optional[ImageDataset]:
    caches = ImageDataset.fetchAll(name = cacheName, include_sessions = 1)
    for cache in caches:
        if cache.count != 0:
            logging.info(">> [Image Augmentation] Cache found!")
            return cache

    return None


def main() -> None:
    taskRun = currentTaskRun()

    outputDatasetName = getOutputDatasetName(taskRun)
    if taskRun.parameters["useCache"]:
        cache = getCache(outputDatasetName.split("-")[1])

        if cache is not None:
            taskRun.submitOutput("outputDataset", cache)
            return

    imagesDataset = taskRun.dataset
    imagesDataset: ImageDataset
    imagesDataset.download()

    backgroundDataset = taskRun.parameters["backgroundDataset"]
    backgroundDataset: ImageDataset
    backgroundDataset.download()

    outputDataset = ImageDataset.createDataset(outputDatasetName, taskRun.projectId)
    if outputDataset is None:
        raise ValueError(">> [Image Stitching] Failed to create output dataset")

    outputDataset.saveClasses(imagesDataset.classes)

    for imageSample in imagesDataset.samples:
        imageSample.unzip()

        logging.info(f">> [Image Stitching] Generating augmented images for {imageSample.name}")
        augmentSample(
            imageSample,
            backgroundDataset,
            taskRun.parameters["rotation"],
            taskRun.parameters["scaling"],
            taskRun.parameters["augmentationsPerImage"],
            imagesDataset.classes,
            taskRun.parameters["documentClass"],
            taskRun.parameters["unwarp"],
            outputDataset
        )

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
