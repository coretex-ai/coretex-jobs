from typing import Optional

from coretex import ComputerVisionSample, BBox

from .results import SampleAccuracyResult, LabelAccuracyResult


def calculateLabelAccuracy(
    groundtruth: BBox,
    prediction: BBox,
    threshold: Optional[float]
) -> float:

    iou = groundtruth.iou(prediction)

    if threshold is None:
        return iou * 100

    if iou > threshold:
        return 100

    return 0


def calculateAccuracy(
    sample: ComputerVisionSample,
    groundtruth: dict[str, BBox],
    prediction: dict[str, BBox],
    threshold: Optional[float]
) -> SampleAccuracyResult:

    labelAccuracies: list[LabelAccuracyResult] = []

    for label, groundtruthBBox in groundtruth.items():
        predictionBBox = prediction.get(label)

        if predictionBBox is None:
            accuracy = 0.0
        else:
            accuracy = calculateLabelAccuracy(groundtruthBBox, predictionBBox, threshold)

        labelAccuracies.append(LabelAccuracyResult(label, accuracy))

    return SampleAccuracyResult(sample.id, sample.name, labelAccuracies)
