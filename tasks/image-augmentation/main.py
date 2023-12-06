from typing import Optional

import logging

import imgaug.augmenters as iaa
from coretex import TaskRun, ImageDataset, currentTaskRun, createDataset
from coretex.utils import hashCacheName

from src.augmentation import augmentImage


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


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    outputDatasetName = getOutputDatasetName(taskRun)

    cache = getCache(outputDatasetName.split("-")[1], taskRun.dataset.count * taskRun.parameters["numOfImages"])
    if cache is not None:
        taskRun.submitOutput("outputDataset", cache)
        return

    dataset = taskRun.dataset
    dataset.download()

    with createDataset(ImageDataset, outputDatasetName, taskRun.projectId) as outputDataset:
        outputDataset.saveClasses(dataset.classes)

        flipH = taskRun.parameters["flipHorizontalPrc"]
        affine = taskRun.parameters["affine"]
        noise = taskRun.parameters["noise"]
        blur = taskRun.parameters["blurPercentage"]
        crop = taskRun.parameters["crop"]
        contrast = taskRun.parameters["contrast"]

        firstAugmenters: list[iaa.Augmenter] = []

        if flipH is not None:
            firstAugmenters.append(iaa.Fliplr(flipH))

        if affine:
            firstAugmenters.append(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            ))

        if crop is not None:
            firstAugmenters.append(iaa.Crop(percent=(0, crop)))

        secondAugmenters: list[iaa.Augmenter] = []

        if noise is not None:
            secondAugmenters.append(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, noise*255), per_channel=0.5))

        if blur is not None:
            secondAugmenters.append(iaa.Sometimes(
                blur,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ))

        if contrast:
            secondAugmenters.append(iaa.LinearContrast((0.75, 1.5)))

        firstPipeline = iaa.Sequential(firstAugmenters)
        secondPipeline = iaa.Sequential(secondAugmenters)

        for sample in dataset.samples:
            logging.info(f">> [Image Augmentation] Performing augmentation on image {sample.name}")

            augmentImage(
                firstPipeline,
                secondPipeline,
                sample,
                taskRun.parameters["numOfImages"],
                outputDataset
            )

        taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
