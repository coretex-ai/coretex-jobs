from pathlib import Path
from typing import Optional

import logging
import random
import json
import yaml

from coretex import ImageDataset, ImageSample, CoretexImageAnnotation, ImageDatasetClasses


def isValidationSplitValid(validationSplit: float, datasetSize: int) -> bool:
    if not 0 <= validationSplit < 1:
        logging.error(f">> [Image Segmentation] validationSplit parameter ({validationSplit}) must be between 0 and 1")
        return False

    minSamplesForSplit = int(1 / min(validationSplit, 1 - validationSplit))
    if datasetSize < minSamplesForSplit:
        logging.error(f">> [Image Segmentation] Dataset is too small ({datasetSize}) for validationSplit parameter ({validationSplit}). Minimum number of samples is {minSamplesForSplit}")
        return False

    return True


def createYamlFile(
    datasetPath: Path,
    trainDatasetPath: Path,
    validDatasetPath: Path,
    classes: ImageDatasetClasses,
    path: Path
) -> None:

    classesCategorical: dict[int, str] = {}
    for index, clazz in enumerate(classes):
        classesCategorical[index] = clazz.label

    data = {
        "path": str(datasetPath.absolute()),
        "train": str(trainDatasetPath.relative_to(datasetPath)),
        "val": str(validDatasetPath.relative_to(datasetPath)),
        "names": classesCategorical
    }

    with path.open("w") as yamlFile:
        yaml.safe_dump(data, yamlFile, default_flow_style = False, sort_keys = False)


def loadAnnotation(sample: ImageSample) -> Optional[CoretexImageAnnotation]:
    if not sample.annotationPath.exists():
        return None

    with sample.annotationPath.open("r") as file:
        return CoretexImageAnnotation.decode(json.load(file))


def saveYoloAnnotation(sample: ImageSample, classes: ImageDatasetClasses, path: Path) -> None:
    sample.unzip()

    destinationPath = path.joinpath(str(sample.id)).with_suffix(sample.imagePath.suffix)
    sample.imagePath.link_to(destinationPath)

    annotation = loadAnnotation(sample)
    if annotation is None:
        return None

    with destinationPath.with_suffix(".txt").open("w") as annotationTxtFile:
        for instance in annotation.instances:
            labelId = classes.labelIdForClassId(instance.classId)

            if labelId is None:
                continue

            for segmentation in instance.segmentations:
                width = annotation.width
                height = annotation.height
                normalizedSegmentation = [num / width if idx % 2 == 0 else num / height for idx, num in enumerate(segmentation)]

                annotationTxtFile.write(str(labelId))
                for coordinate in normalizedSegmentation:
                    # The annotation for a single segmentation should look like this: "x1 y1 x2 y2 x3 y3 x4 y4..." where the numbers represent the coordinates of the segmentation polygon.
                    annotationTxtFile.write(f" {coordinate}")

                annotationTxtFile.write("\n")


def prepareDataset(dataset: ImageDataset, destination: Path, validationPct: float) -> tuple[Path, Path]:
    validCount = int(dataset.count * validationPct)
    trainCount = dataset.count - validCount

    samples = dataset.samples.copy()
    random.shuffle(samples)

    trainDatasetPath = destination / "train"
    trainDatasetPath.mkdir()

    validDatasetPath = destination / "val"
    validDatasetPath.mkdir()

    trainSamples = samples[:trainCount]
    validSamples = samples[trainCount:]

    for sample in trainSamples:
        saveYoloAnnotation(sample, dataset.classes, trainDatasetPath)

    for sample in validSamples:
        saveYoloAnnotation(sample, dataset.classes, validDatasetPath)

    return trainDatasetPath, validDatasetPath
