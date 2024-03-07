from coretex import CoretexSegmentationInstance, CoretexImageAnnotation, ImageDatasetClass

import numpy as np


def getClassAnnotation(annotation: CoretexImageAnnotation, class_: ImageDatasetClass) -> CoretexSegmentationInstance:
    for instance in annotation.instances:
        if instance.classId in class_.classIds:
            return instance

    raise ValueError(f"Failed to find \"{class_.label}\" annotation")


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
