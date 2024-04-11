from typing import Optional
from pathlib import Path

import logging

from coretex import ImageDataset, ImageDatasetClasses, ImageDatasetClass, BBox, ImageSample
from ultralytics import YOLO
from ultralytics.engine.results import Results

import matplotlib.pyplot as plt
import matplotlib.patches as pth


def classByLabelId(labelId: int, classes: ImageDatasetClasses) -> Optional[ImageDatasetClass]:
    return classes.classByLabel(classes.labels[labelId])


def processResult(result: Results, classes: list[ImageDatasetClasses], savePath: Path):
    fig = plt.figure(num = 1, clear = True)
    plt.imshow(result.orig_img)

    if result.boxes is not None:
        for minX, minY, maxX, maxY, confidence, labelId in result.boxes.data:
            box = BBox.create(minX, minY, maxX, maxY)

            clazz = classByLabelId(int(labelId), classes)
            if clazz is None:
                continue

            fig.gca().add_patch(pth.Rectangle(
                (float(box.minX), float(box.minY)),
                float(box.width),
                float(box.height),
                linewidth = 3,
                edgecolor = clazz.color,
                facecolor = "none"
            ))

    plt.savefig(savePath)


def isSampleValid(sample: ImageSample) -> bool:
    try:
        for instance in sample.load().annotation.instances:
            if any([len(segmentation) < 6 for segmentation in instance.segmentations]):
                return False
    except Exception as e:
        logging.debug(f"Falied to load sample annotation data for {sample.name}, ID: {sample.id}. Error: {e}")
        return False

    return True


def predictBatch(model: YOLO, dataset: ImageDataset, startIdx: int, endIdx: int, resultPath: Path):
    batch = [sample for sample in dataset.samples[startIdx:endIdx] if isSampleValid(sample)]

    results: Results = model.predict([sample.imagePath for sample in batch], save = True, project = "./results")
    for sample, result in zip(batch, results):
        processResult(result, dataset.classes, resultPath / f"{sample.name}.png")


def run(model: YOLO, dataset: ImageDataset, resultPath: Path, batchSize: int) -> None:
    for i in range(0, dataset.count - (dataset.count % batchSize), batchSize):
        predictBatch(model, dataset, i, i + batchSize, resultPath)

    # Remainder
    predictBatch(model, dataset, dataset.count - (dataset.count % batchSize), dataset.count, resultPath)
