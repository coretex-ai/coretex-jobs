from typing import Optional
from pathlib import Path

import logging

import imgaug.augmenters as iaa
import imageio.v3 as imageio

from coretex import TaskRun, ImageDataset, ImageSample, folder_manager, currentTaskRun
from coretex.utils import hashCacheName


def getOutputDatasetName(taskRun: TaskRun) -> str:
    relevantParams = taskRun.parameters.copy()

    relevantParams["dataset"] = relevantParams["dataset"].id
    relevantParams.pop("useCache")
    relevantParams.pop("outputDataset")

    return hashCacheName(f"{taskRun.id}-AugImg", ".".join(str(relevantParams.values())))


def getCache(cacheName: str) -> Optional[ImageDataset]:
    caches = ImageDataset.fetchAll(name = cacheName, include_sessions = 1)
    for cache in caches:
        if cache.count != 0:
            logging.info(">> [Image Augmentation] Cache found!")
            return cache

    return None


def augmentImage(augmenter: iaa.Sequential, sample: ImageSample, numOfImages: int, outputDir: Path, outputDataset: ImageDataset) -> None:
    sample.unzip()

    image = imageio.imread(sample.imagePath)
    for i in range(numOfImages):
        augmentedImage = augmenter.augment_image(image)
        outputPath = outputDir / f"{sample.name}-{i}.jpg"
        imageio.imwrite(outputPath, augmentedImage)

        if ImageSample.createImageSample(outputDataset.id, outputPath) is None:
            logging.error(f">> [Image Augmentation] {outputPath.name} failed to uplaod")
        else:
            logging.info(f">> [Image Augmentation] Uploaded {outputPath.name} to coretex")


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    outputDatasetName = getOutputDatasetName(taskRun)
    if taskRun.parameters["useCache"]:
        cache = getCache(outputDatasetName.split("-")[1])

        if cache is not None:
            taskRun.submitOutput("outputDataset", cache)

    dataset = taskRun.dataset
    dataset.download()

    outputDataset = ImageDataset().createDataset(outputDatasetName, taskRun.projectId)
    if outputDataset is None:
        raise ValueError(">> [Image Augmentation] Failed to create output dataset")

    outputDir = folder_manager.createTempFolder("augmentedImages")

    flipH = taskRun.parameters["flipHorizontalPrc"]
    flipV = taskRun.parameters["flipVerticalPrc"]
    affine = taskRun.parameters["affine"]
    noise = taskRun.parameters["noise"]
    blur = taskRun.parameters["blurPrc"]
    crop = taskRun.parameters["crop"]
    contrast = taskRun.parameters["contrast"]

    augmenters: list[iaa.Augmenter] = []

    if flipH is not None:
        augmenters.append(iaa.Fliplr(flipH))

    if flipV is not None:
        augmenters.append(iaa.Flipud(flipV))

    if affine:
        augmenters.append(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        ))

    if crop is not None:
        augmenters.append(iaa.Crop(percent=(0, crop)))

    if noise is not None:
        augmenters.append(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, noise*255), per_channel=0.5))

    if blur is not None:
        augmenters.append(iaa.Sometimes(
            blur,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ))

    if contrast:
        augmenters.append(iaa.LinearContrast((0.75, 1.5)))

    augmentationPipeline = iaa.Sequential(augmenters)

    for sample in dataset.samples:
        logging.info(f">> [Image Augmentation] Performing augmentation on image {sample.name}")
        augmentImage(augmentationPipeline, sample, taskRun.parameters["numOfImages"], outputDir, outputDataset)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
