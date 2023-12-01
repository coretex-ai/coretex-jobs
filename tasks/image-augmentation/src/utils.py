import logging

from numpy import ndarray

import imageio.v3 as imageio

from coretex import CoretexImageAnnotation, ImageDataset, folder_manager, ImageSample


def uploadAugmentedImage(
    imageName: str,
    augmentedImage: ndarray,
    annotation: CoretexImageAnnotation,
    outputDataset: ImageDataset
) -> None:

    imagePath = folder_manager.temp / f"{imageName}.jpeg"
    imageio.imwrite(imagePath, augmentedImage)

    augmentedSample = ImageSample.createImageSample(outputDataset.id, imagePath)
    if augmentedSample is None:
        logging.error(f">> [Image Augmentation] Failed to upload sample {imagePath}")
        return

    augmentedSample.download()
    augmentedSample.unzip()
    if not augmentedSample.saveAnnotation(annotation):
        logging.error(f">> [Image Augmentation] Failed to update sample annotation {imagePath}")
