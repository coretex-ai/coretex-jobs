from coretex import AnnotatedImageSampleData, ImageDatasetClasses, CoretexImageAnnotation, CoretexSegmentationInstance, BBox

import cv2
import numpy as np

from . import rect_extractor, utils


def extractDocumentData(data: AnnotatedImageSampleData, classes: ImageDatasetClasses) -> tuple[np.ndarray, CoretexImageAnnotation]:
    image = data.image
    annotation = data.annotation

    if annotation is None:
        raise ValueError("Missing annotation")

    documentInstance = utils.getDocumentInstance(annotation, classes)
    mask = documentInstance.extractBinaryMask(annotation.width, annotation.height)

    rect = rect_extractor.extractRectangle(mask)

    documentSegmentation = [0, 0, rect.width, 0, rect.width, rect.height, 0, rect.height]
    transformed = np.array(documentSegmentation, dtype = np.float32).reshape((-1, 2))
    transformMatrix = cv2.getPerspectiveTransform(rect.numpy(), transformed)
    maskedImage = cv2.warpPerspective(image, transformMatrix, (rect.width, rect.height))

    transformedInstances: list[CoretexSegmentationInstance] = []

    documentClass = classes.classByLabel("document")
    if documentClass is None:
        raise ValueError("Missing document class")

    transformedInstances.append(CoretexSegmentationInstance.create(
        documentClass.classIds[0],
        BBox.fromPoly(documentSegmentation),
        [documentSegmentation]
    ))

    for instance in annotation.instances:
        if instance.classId in documentClass.classIds:
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
    return maskedImage, transformedAnnotation
