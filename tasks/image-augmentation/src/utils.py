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

    if not augmentedSample.saveAnnotation(annotation):
        logging.error(f">> [Image Augmentation] Failed to update sample annotation {imagePath}")


def copySample(sample: ImageSample, dataset: ImageDataset) -> None:
    sample.unzip()

    copy = ImageSample.createImageSample(dataset.id, sample.imagePath)
    if copy is None:
        logging.error(f"\tFailed to copy sample \"{sample.name}\"")
        return

    annotation = sample.load().annotation
    if annotation is not None:
        if not copy.saveAnnotation(annotation):
            logging.error("\tFailed to copy sample annotation, deleting...")

            if not copy.delete():
                logging.error("\tFailed to delete sample")
