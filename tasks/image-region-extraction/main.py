from typing import Optional
from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future

import logging
import os

from PIL import Image
from coretex import ImageDataset, ImageSample, ImageDatasetClasses, CoretexSegmentationInstance, \
    folder_manager, currentTaskRun, createDataset

from src.extractor import extractParent, getParentInstance, extractRegion


def didGenerateSample(datasetId: int, future: Future[list[Path]]) -> None:
    try:
        imagePaths = future.result()
        for imagePath in imagePaths:
            generatedSample = ImageSample.createImageSample(datasetId, imagePath)
            if generatedSample is not None:
                logging.info(f">> [RegionExtraction] Generated sample \"{generatedSample.name}\"")
            else:
                logging.error(f">> [RegionExtraction] Failed to create sample from \"{imagePath}\"")
    except BaseException as exception:
        logging.error(f">> [RegionExtraction] Failed to generate sample. Reason: {exception}")
        logging.debug(exception, exc_info = exception)


def processSample(
    sample: ImageSample,
    classes: ImageDatasetClasses,
    parentClassName: Optional[str],
    excludedClasses: Optional[list[str]]
) -> list[Path]:

    extractedImagesDir = folder_manager.createTempFolder(f"{sample.id}")
    sampleData = sample.load()

    annotation = sampleData.annotation
    if annotation is None:
        raise ValueError(f">> [RegionExtraction] CTX sample dataset sample id: {sample.id} image doesn't exist!")

    image = sampleData.image
    if parentClassName is not None:
        parentClass = getParentInstance(annotation, classes, parentClassName)
        parentMask = parentClass.extractBinaryMask(image.shape[1], image.shape[0])

        image, annotation = extractParent(image, parentMask, parentClassName, classes, annotation)

    extractedImages: list[Path] = []
    for i, instance in enumerate(annotation.instances):
        # Do not extract parent class
        if parentClassName is not None and instance.classId == parentClass.classId:
            continue

        # Skip excluded classes
        if excludedClasses is not None and classes.classById(instance.classId).label in excludedClasses:
            continue

        foregroundMask = instance.extractBinaryMask(image.shape[1], image.shape[0])
        extractedImage = extractRegion(image, foregroundMask)

        extractedImagePath = extractedImagesDir / f"{sample.name}-{classes.classById(instance.classId).label}-{i}.png"
        extractedImage.save(extractedImagePath)
        extractedImages.append(extractedImagePath)

    return extractedImages


def main() -> None:
    taskRun = currentTaskRun()

    imagesDataset = taskRun.dataset
    imagesDataset: ImageDataset
    imagesDataset.download()

    with createDataset(ImageDataset, f"{taskRun.id}-ExtractedImages", taskRun.projectId) as outputDataset:
        outputDataset.saveClasses(imagesDataset.classes)

        with ExitStack() as stack:
            executor = ProcessPoolExecutor(max_workers = os.cpu_count())
            stack.enter_context(executor)

            uploader = ThreadPoolExecutor(max_workers = 4)
            stack.enter_context(uploader)

            for imageSample in imagesDataset.samples:
                imageSample.unzip()

                logging.info(f">> [RegionExtraction] Extracting annotated regions for {imageSample.name}")
                future = executor.submit(
                    processSample,
                    imageSample,
                    imagesDataset.classes,
                    taskRun.parameters["parentClass"],
                    taskRun.parameters["excludedClasses"]
                )

                uploader.submit(didGenerateSample, outputDataset.id, future)

        taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
