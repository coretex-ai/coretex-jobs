import logging

from numpy import ndarray

import imageio.v3 as imageio

from coretex import CoretexImageAnnotation, ImageDataset, folder_manager, ImageSample


def uploadAugmentedImage(
    imageName: str,
    augmentedImage: ndarray,
    annotation: CoretexImageAnnotation,
    originalSample: ImageSample,
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

    try:
        metadata = originalSample.loadMetadata()
        augmentedSample.saveMetadata(metadata)
    except FileNotFoundError:
        logging.info(f">> [Image Augmentation] The metadata for sample \"{originalSample.name}\" was not found")
    except ValueError:
        logging.info(f">> [Image Augmentation] Invalid metadata type for sample \"{originalSample.name}\"")


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

    try:
        metadata = sample.loadMetadata()
        copy.saveMetadata(metadata)
    except FileNotFoundError:
        logging.info(f">> [Image Augmentation] The metadata for sample \"{sample.name}\" was not found")
    except ValueError:
        logging.info(f">> [Image Augmentation] Invalid metadata type for sample \"{sample.name}\"")
