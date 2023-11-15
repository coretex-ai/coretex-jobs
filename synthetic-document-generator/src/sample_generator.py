from pathlib import Path

import random
import uuid

from coretex import ComputerVisionSample, CoretexImageAnnotation, CoretexSegmentationInstance, ImageDatasetClasses, folder_manager
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from src import document_extractor, utils


def updateAnnotationPosition(
    annotation: CoretexImageAnnotation,
    imageWidth: int,
    imageHeight: int,
    documentWidth: int,
    documentHeight: int,
    documentTopX: int,
    documentTopY: int
) -> CoretexImageAnnotation:

    resizedInstances: list[CoretexSegmentationInstance] = []

    for instance in annotation.instances:
        resizedInstance = utils.resizeInstance(instance, annotation.width, annotation.height, documentWidth, documentHeight)
        utils.offsetSegmentations(resizedInstance, documentTopX, documentTopY)

        resizedInstances.append(resizedInstance)

    return CoretexImageAnnotation.create(annotation.name, imageWidth, imageHeight, resizedInstances)


def generateSample(
    sample: ComputerVisionSample,
    backgroundImage: np.ndarray,
    classes: ImageDatasetClasses,
    minDocumentSize: float,
    maxDocumentSize: float
) -> tuple[Path, CoretexImageAnnotation]:

    data = sample.load()
    documentImage, transformedAnnotation = document_extractor.extractDocumentData(data, classes)

    pBackgroundImage = Image.fromarray(backgroundImage)
    pDocumentImage = Image.fromarray(documentImage)

    # Resize document image
    documentWidth = int(pBackgroundImage.width * random.uniform(minDocumentSize, maxDocumentSize))
    documentHeight = int(documentWidth * (pDocumentImage.height / pDocumentImage.width))

    pDocumentImage = pDocumentImage.resize((documentWidth, documentHeight), Image.Resampling.LANCZOS)

    # Paste document image to background image
    documentTopX = random.randint(0, pBackgroundImage.width - documentWidth)
    documentTopY = random.randint(0, pBackgroundImage.height - documentHeight)
    pBackgroundImage.paste(pDocumentImage, (documentTopX, documentTopY))

    updatedAnnotation = updateAnnotationPosition(
        transformedAnnotation,
        pBackgroundImage.width,
        pBackgroundImage.height,
        documentWidth,
        documentHeight,
        documentTopX,
        documentTopY
    )

    # plt.imshow(pBackgroundImage)
    # for instance in updatedAnnotation.instances:
    #     for segmentation in instance.segmentations:
    #         plt.plot(
    #             [p for i, p in enumerate(segmentation) if i % 2 == 0],
    #             [p for i, p in enumerate(segmentation) if i % 2 != 0],
    #             color = classes.classById(instance.classId).color
    #         )
    # plt.show()

    identifier = uuid.uuid4()
    imagePath = folder_manager.temp / f"{identifier}.jpeg"
    pBackgroundImage.save(imagePath)

    return imagePath, updatedAnnotation
