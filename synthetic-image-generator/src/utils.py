import logging

from PIL.Image import Image as PILImage

from coretex import ImageSample, ImageDataset, CoretexImageAnnotation, folder_manager


ANNOTATION_NAME = "annotations.json"


def uploadAugmentedImage(
    imageName: str,
    augmentedImage: PILImage,
    annotation: CoretexImageAnnotation,
    outputDataset: ImageDataset
) -> None:

    imagePath = folder_manager.temp / f"{imageName}.jpeg"
    augmentedImage.save(imagePath)

    augmentedSample = ImageSample.createImageSample(outputDataset.id, imagePath)
    if augmentedSample is None:
        logging.error(f">> [Image Stitching] Failed to upload sample {imagePath}")
        return

    augmentedSample.download()
    augmentedSample.unzip()
    if not augmentedSample.saveAnnotation(annotation):
        logging.error(f">> [Image Stitching] Failed to update sample annotation {imagePath}")

    imagePath.unlink(missing_ok = True)
