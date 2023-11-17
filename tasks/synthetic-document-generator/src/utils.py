from coretex import CoretexSegmentationInstance, CoretexImageAnnotation, ImageDatasetClasses, BBox

import numpy as np


def getDocumentInstance(annotation: CoretexImageAnnotation, classes: ImageDatasetClasses) -> CoretexSegmentationInstance:
    documentClass = classes.classByLabel("document")
    if documentClass is None:
        raise ValueError("Missing document class")

    for instance in annotation.instances:
        if instance.classId in documentClass.classIds:
            return instance

    raise ValueError("Failed to find document annotation")


def offsetSegmentations(instance: CoretexSegmentationInstance, offsetX: int, offsetY: int) -> None:
    for segmentation in instance.segmentations:
        for i, p in enumerate(segmentation):
            if i % 2 == 0:
                segmentation[i] = p + offsetX
            else:
                segmentation[i] = p + offsetY

    instance.bbox = BBox.fromPoly([e for sublist in instance.segmentations for e in sublist])


def warpPerspectivePoint(matrix: np.ndarray, x: int, y: int) -> tuple[int, int]:
    px = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / ((matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]))
    py = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / ((matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]))

    return abs(px), abs(py)


def warpPerspectivePoly(transformMatrix: np.ndarray, points: list[int]) -> list[int]:
    result: list[int] = []

    for i in range(1, len(points), 2):
        x, y = warpPerspectivePoint(transformMatrix, points[i - 1], points[i])
        result.extend([x, y])

    return result


def toPoly(segmentation: list[int]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []

    for index in range(0, len(segmentation) - 1, 2):
        points.append((segmentation[index], segmentation[index + 1]))

    return points


def resizePolygon(polygon: list[int], oldWidth: int, oldHeight: int, newWidth: int, newHeight: int) -> list[int]:
    resized: list[int] = []

    for i, value in enumerate(polygon):
        if i % 2 == 0:
            resized.append(int(value / oldWidth * newWidth))
        else:
            resized.append(int(value / oldHeight * newHeight))

    return resized


def resizeInstance(
    instance: CoretexSegmentationInstance,
    oldWidth: int,
    oldHeight: int,
    newWidth: int,
    newHeight: int
) -> CoretexSegmentationInstance:

    resizedSegmentations: list[list[int]] = []

    for segmentation in instance.segmentations:
        resizedSegmentations.append(resizePolygon(segmentation, oldWidth, oldHeight, newWidth, newHeight))

    return CoretexSegmentationInstance.create(
        instance.classId,
        BBox.fromPoly([e for sublist in resizedSegmentations for e in sublist]),
        resizedSegmentations
    )
