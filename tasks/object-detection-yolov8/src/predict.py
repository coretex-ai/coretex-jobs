from typing import Optional
from pathlib import Path

from coretex import ComputerVisionDataset, ImageDatasetClasses, ImageDatasetClass, BBox, ComputerVisionSample
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


def isSampleValid(sample: ComputerVisionSample) -> bool:
    for instance in sample.load().annotation.instances:
        if any([len(segmentation) < 6 for segmentation in instance.segmentations]):
            return False

    return True


def run(model: YOLO, dataset: ComputerVisionDataset, resultPath: Path, batchSize: int) -> None:
    for i in range(0, len(dataset.samples), batchSize):
        if (dataset.count - i) < batchSize:
            batchSize = (dataset.count - i)

        batch = [dataset.samples[index].imagePath for index in range(i, i + batchSize) if isSampleValid(dataset.samples[index])]
        results: Results = model.predict(batch, save = True, project = "./results")
        [processResult(result, dataset.classes, resultPath / f"{dataset.samples[i + j].name}.png") for j, result in enumerate(results)]
