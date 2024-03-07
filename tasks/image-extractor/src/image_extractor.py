from typing import Generator, Optional

from coretex import CoretexImageAnnotation, CoretexSegmentationInstance, BBox, ImageDatasetClass
from PIL import Image

import cv2
import numpy as np

from . import rect_extractor, utils
from .rect import Rect


def warpPerspective(
    image: np.ndarray,
    instance: CoretexSegmentationInstance
) -> tuple[Rect, list[int], np.ndarray, np.ndarray]:

    mask = instance.extractBinaryMask(image.shape[1], image.shape[0])

    rect = rect_extractor.extractRectangle(mask)
    segmentation = [0, 0, rect.width, 0, rect.width, rect.height, 0, rect.height]

    transformed = np.array(segmentation, dtype = np.float32).reshape((-1, 2))
    transformMatrix = cv2.getPerspectiveTransform(rect.numpy(), transformed)
    transformedImage = cv2.warpPerspective(image, transformMatrix, (rect.width, rect.height))

    return rect, segmentation, transformMatrix, transformedImage


def extractWithParent(
    image: np.ndarray,
    annotation: CoretexImageAnnotation,
    parentClass: ImageDatasetClass
) -> tuple[np.ndarray, CoretexImageAnnotation]:

    parentInstance = utils.getClassAnnotation(annotation, parentClass)
    rect, parentSegmentation, transformMatrix, transformedImage = warpPerspective(image, parentInstance)

    transformedInstances = [
        CoretexSegmentationInstance.create(
            parentClass.classIds[0],
            BBox.fromPoly(parentSegmentation),
            [parentSegmentation]
        )
    ]

    for instance in annotation.instances:
        if instance.classId in parentClass.classIds:
            continue

        transformedSegmentations = [
            utils.warpPerspectivePoly(transformMatrix, segmentation)
            for segmentation in instance.segmentations
        ]
        flattenedSegmentations = [e for segmentation in transformedSegmentations for e in segmentation]

        transformedInstances.append(CoretexSegmentationInstance.create(
            instance.classId,
            BBox.fromPoly(flattenedSegmentations),
            transformedSegmentations
        ))

    transformedAnnotation = CoretexImageAnnotation.create(annotation.name, rect.width, rect.height, transformedInstances)
    return transformedImage, transformedAnnotation


def extract(
    image: np.ndarray,
    annotation: CoretexImageAnnotation
) -> Generator[tuple[np.ndarray, None], None, None]:

    def extractRegion(image: np.ndarray, segmentationMask: np.ndarray) -> np.ndarray:
        rgbaImage = Image.fromarray(image).convert("RGBA")

        segmentedImage = np.asarray(rgbaImage) * segmentationMask[..., None]
        segmentedImage = Image.fromarray(segmentedImage)

        alpha = segmentedImage.getchannel("A")
        bbox = alpha.getbbox()

        return np.asarray(segmentedImage.crop(bbox))

    for instance in annotation.instances:
        mask = instance.extractBinaryMask(annotation.width, annotation.height)
        yield extractRegion(image, mask), None


def extractImages(
    image: np.ndarray,
    annotation: CoretexImageAnnotation,
    parentClass: Optional[ImageDatasetClass] = None
) -> Generator[tuple[np.ndarray, Optional[CoretexImageAnnotation]], None, None]:

    if parentClass is not None:
        yield extractWithParent(image, annotation, parentClass)
    else:
        yield from extract(image, annotation)
