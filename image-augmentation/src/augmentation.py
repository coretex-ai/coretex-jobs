from pathlib import Path

import logging

from PIL import Image

import cv2
import numpy as np
import imgaug.augmenters as iaa
import imageio.v3 as imageio
import matplotlib.pyplot as plt

from coretex import TaskRun, ImageDataset, ImageSample


def mask2poly(mask: np.ndarray) -> list[int]:
    cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


def augmentImage(
    firstPipeline: iaa.Sequential,
    secondPipeline: iaa.Sequential,
    sample: ImageSample,
    numOfImages: int,
    outputDir: Path,
    outputDataset: ImageDataset
) -> None:

    sample.unzip()

    image = imageio.imread(sample.imagePath)
    sampleData = sample.load()

    for i in range(numOfImages):
        fig, axes = plt.subplots(1, 3)

        secondPipeline_ = secondPipeline.localize_random_state()
        secondPipeline_ = secondPipeline_.to_deterministic()

        augmentedImage = secondPipeline_.augment_image(image)
        axes[0].imshow(augmentedImage)

        for instance in sampleData.annotation.instances:
            mask = instance.extractSegmentationMask(sampleData.annotation.width, sampleData.annotation.height)
            axes[2].imshow(mask)
            mask = np.repeat(mask[..., None] * 255, 3, axis = -1)
            augmentedMask = secondPipeline_.augment_image(mask)
            augmentedMask = (np.average(augmentedMask, axis=-1) > 128).astype(int)
            axes[1].imshow(augmentedMask)

            plt.show()
            plt.close()

            Image.fromarray(augmentedMask).show()

        outputPath = outputDir / f"{sample.name}-{i}.jpg"
        imageio.imwrite(outputPath, augmentedImage)

        if ImageSample.createImageSample(outputDataset.id, outputPath) is None:
            logging.error(f">> [Image Augmentation] {outputPath.name} failed to uplaod")
        else:
            logging.info(f">> [Image Augmentation] Uploaded {outputPath.name} to coretex")