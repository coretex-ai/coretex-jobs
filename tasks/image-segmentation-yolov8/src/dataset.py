from pathlib import Path
from typing import Optional

import logging
import random
import json
import yaml

from coretex import ImageDataset, ImageSample, CoretexImageAnnotation, ImageDatasetClasses


def isValidationSplitValid(validationSplit: float, datasetSize: int) -> bool:
    if not 0 <= validationSplit < 1:
        logging.error(f">> [ObjectDetection] validationSplit parameter ({validationSplit}) must be between 0 and 1")
        return False

    minSamplesForSplit = int(1 / min(validationSplit, 1 - validationSplit))
    if datasetSize < minSamplesForSplit:
        logging.error(f">> [ObjectDetection] Dataset is too small ({datasetSize}) for validationSplit parameter ({validationSplit}). Minimum number of samples is {minSamplesForSplit}")
        return False

    return True


def createYamlFile(
    datasetPath: Path,
    trainDatasetPath: Path,
    validDatasetPath: Path,
    classes: ImageDatasetClasses,
    location: Path
) -> None:

    classesIds: dict[int, str] = {}
    for index, clazz in enumerate(classes):
        classesIds[index] = clazz.label

    data = {
        "path": str(datasetPath.absolute()),
        "train": str(trainDatasetPath.relative_to(datasetPath)),
        "val": str(validDatasetPath.relative_to(datasetPath)),
        "names": classesIds
    }

    with open(location, "w") as yamlFIle:
        yaml.dump(data, yamlFIle, default_flow_style = False, sort_keys = False)


def loadAnnotation(sample: ImageSample) -> Optional[CoretexImageAnnotation]:
    if not sample.annotationPath.exists():
        return None

    with sample.annotationPath.open("r") as file:
        return CoretexImageAnnotation.decode(json.load(file))


def addImagePathForYaml(sample: ImageSample, imagesPath: Path) -> None:
    sample.unzip()
    sample.imagePath.link_to(imagesPath / f"{sample.id}{sample.imagePath.suffix}")


def addAnnotation(sample: ImageSample, classes: ImageDatasetClasses, location: Path) -> None:
    sample.unzip()

    annotation = loadAnnotation(sample)
    if annotation is not None:
        normalizedAnnotations: list[tuple[int, list[float]]] = []

        for instance in annotation.instances:
            labelId = classes.labelIdForClassId(instance.classId)

            if labelId is None:
                continue

            for segmentation in instance.segmentations:
                w = annotation.width
                h = annotation.height
                normalizedSegmentation = [num / w if idx % 2 == 0 else num / h for idx, num in enumerate(segmentation)]

                normalizedAnnotation = labelId, normalizedSegmentation

                normalizedAnnotations.append(normalizedAnnotation)

        with open(location / f"{sample.id}.txt", "w") as txtFile:
            for segmentation in normalizedAnnotations:
                txtFile.write(str(segmentation[0]))
                coordinates = segmentation[1]
                for coord in coordinates:
                    txtFile.write(f" {coord}")

                txtFile.write("\n")


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
        addImagePathForYaml(sample, trainDatasetPath)
        addAnnotation(sample, dataset.classes, trainDatasetPath)

    for sample in validSamples:
        addImagePathForYaml(sample, validDatasetPath)
        addAnnotation(sample, dataset.classes, validDatasetPath)

    return trainDatasetPath, validDatasetPath
