from concurrent.futures import ProcessPoolExecutor

import logging
import os

from PIL import Image
from coretex import ImageDataset, ImageSample, ImageDatasetClasses, CoretexSegmentationInstance, \
    folder_manager, currentTaskRun, createDataset

from src.extractor import extractParent, getParentInstance, extractRegion


def processSample(
    sample: ImageSample,
    classes: ImageDatasetClasses,
    parentClassName: str,
    excludedClasses: list[str],
    outputDataset: ImageDataset
) -> tuple[Image.Image, list[CoretexSegmentationInstance]]:

    extractedImagesDir = folder_manager.createTempFolder(f"{sample.id}")
    sampleData = sample.load()

    annotation = sampleData.annotation
    if annotation is None:
        raise RuntimeError(f"CTX sample dataset sample id: {sample.id} image doesn't exist!")

    parentClass = getParentInstance(annotation, classes, parentClassName)
    parentMask = parentClass.extractBinaryMask(sampleData.image.shape[1], sampleData.image.shape[0])

    parentImage, newAnnotation = extractParent(sampleData.image, parentMask, parentClassName, classes, annotation)
    if newAnnotation is None:
        return None

    for i, instance in enumerate(newAnnotation.instances):
        # Do not extract parent class
        if instance.classId == parentClass.classId:
            continue

        # Skip excluded classes
        if excludedClasses is not None and classes.classById(instance.classId).label in excludedClasses:
            continue

        foregroundMask = instance.extractBinaryMask(parentImage.shape[1], parentImage.shape[0])
        extractedImage = extractRegion(parentImage, foregroundMask)

        extractedImagePath = extractedImagesDir / f"{sample.name}-{classes.classById(instance.classId).label}-{i}.png"
        extractedImage.save(extractedImagePath)
        ImageSample.createImageSample(outputDataset.id, extractedImagePath)


def main() -> None:
    taskRun = currentTaskRun()

    imagesDataset = taskRun.dataset
    imagesDataset: ImageDataset
    imagesDataset.download()

    with createDataset(ImageDataset, f"{taskRun.id}-ExtractedImages", taskRun.projectId) as outputDataset:
        outputDataset.saveClasses(imagesDataset.classes)

        with ProcessPoolExecutor(max_workers = os.cpu_count()) as executor:
            for imageSample in imagesDataset.samples:
                imageSample.unzip()

                logging.info(f">> [RegionExtraction] Extracting annotated regions for {imageSample.name}")
                executor.submit(
                    processSample,
                    imageSample,
                    imagesDataset.classes,
                    taskRun.parameters["parentClass"],
                    taskRun.parameters["excludedClasses"],
                    outputDataset
                )

        taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
