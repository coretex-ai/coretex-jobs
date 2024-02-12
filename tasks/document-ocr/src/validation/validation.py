from typing import Optional

from coretex import ComputerVisionSample, BBox

from .results import SampleAccuracyResult, LabelAccuracyResult


def calculateLabelAccuracy(
    groundtruth: BBox,
    prediction: BBox,
    threshold: Optional[float]
) -> tuple[float, float]:

    iou = groundtruth.iou(prediction)

    if threshold is None:
        accuracy = iou * 100
    else:
        accuracy = 100 if iou >= threshold else 0

    return accuracy, iou


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
            accuracy, iou = 0.0, 0.0
        else:
            accuracy, iou = calculateLabelAccuracy(groundtruthBBox, predictionBBox, threshold)

        labelAccuracies.append(LabelAccuracyResult(label, accuracy, iou))

    return SampleAccuracyResult(sample.id, sample.name, labelAccuracies)
