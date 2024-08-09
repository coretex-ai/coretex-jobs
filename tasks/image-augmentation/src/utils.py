from typing import Optional

import logging

from numpy import ndarray

import imageio.v3 as imageio

from coretex import CoretexImageAnnotation, ImageDataset, folder_manager, ImageSample


def uploadAugmentedImage(
    imageName: str,
    augmentedImage: ndarray,
    annotation: Optional[CoretexImageAnnotation],
    outputDataset: ImageDataset
) -> None:

    imagePath = folder_manager.temp / imageName
    imageio.imwrite(imagePath, augmentedImage)

    try:
        augmentedSample = outputDataset.add(imagePath)
    except BaseException as ex:
        logging.error(f">> [Image Augmentation] Failed to upload sample {imagePath} - \"{ex}\"")
        return

    if annotation is not None:
        if not augmentedSample.saveAnnotation(annotation):
            logging.error(f">> [Image Augmentation] Failed to update sample annotation {imagePath}")


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
