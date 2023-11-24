from typing import Optional

from numpy import ndarray

import numpy as np

from coretex import CoretexSegmentationInstance, BBox

from .utils import Rect


SegmentationType = list[int]


def centerSegmentations(
    segmentations: list[SegmentationType],
    newCentroid: tuple[int, int],
    oldCentroid: tuple[int, int]
    ) -> list[SegmentationType]:

    newCenterX, newCenterY = newCentroid
    oldCenterX, oldCenterY = oldCentroid

    modifiedSegmentations: list[list[int]] = []

    for segmentation in segmentations:
        modifiedSegmentation: list[int] = []

        for i in range(0, len(segmentation), 2):
            x = segmentation[i] + (newCenterX - oldCenterX)
            y = segmentation[i+1] + (newCenterY - oldCenterY)

            modifiedSegmentation.append(x)
            modifiedSegmentation.append(y)

        modifiedSegmentations.append(modifiedSegmentation)

    return modifiedSegmentations


def applyAffine(point: tuple[int, int], transformMatrix: ndarray) -> tuple[int, int]:
    columnVector = np.array([[point[0], point[1], 1]]).T
    hm = np.matmul(transformMatrix, columnVector).flatten()
    return (abs(int(hm[0] / hm[2])), abs(int(hm[1] / [hm[2]])))


def affineSegmentation(inputSegmentation: list[int], transformMatrix: np.ndarray) -> SegmentationType:
    outputSegmentation: list[int] = []
    for i in range(0, len(inputSegmentation), 2):
        cornerPoint = (inputSegmentation[i], inputSegmentation[i + 1])
        outputSegmentation.extend(applyAffine(cornerPoint, transformMatrix))

    return outputSegmentation


def getCenter(segmentations: list[SegmentationType]) -> tuple[int, int]:
    segmentation = segmentations[0]
    xs = [segmentation[i] for i in range(0, len(segmentation), 2)]
    ys = [segmentation[i + 1] for i in range(0, len(segmentation), 2)]
    center = (0.5 * (max(xs) - min(xs)) + min(xs), 0.5 * (max(ys) - min(ys)) + min(ys))

    return center


def transformAnnotation(
    instance: CoretexSegmentationInstance,
    transformMatrix: Optional[np.ndarray],
    scale: float,
    centroid: tuple[int, int],
    angle: int,
    unwarp: bool,
    mask: ndarray
) -> CoretexSegmentationInstance:

    segmentations = instance.segmentations
    if unwarp and transformMatrix is not None:
        points = Rect.extractRectangle(mask).numpy().flatten()
        segmentations = [np.append(points, [points[0], points[1]]).tolist()]
        segmentations = [affineSegmentation(sublist, transformMatrix) for sublist in segmentations]

    segmentationsScaled = [[value * scale for value in sublist] for sublist in segmentations]
    segmentationsFlattened = [sample for sublist in segmentationsScaled for sample in sublist]

    augmentedInstance = CoretexSegmentationInstance.create(
        instance.classId,
        BBox.fromPoly(segmentationsFlattened),
        segmentationsScaled
    )

    augmentedInstance.rotateSegmentations(angle)
    augmentedInstance.segmentations = centerSegmentations(
        augmentedInstance.segmentations,
        newCentroid = centroid,
        oldCentroid = getCenter(augmentedInstance.segmentations)
    )

    return augmentedInstance
