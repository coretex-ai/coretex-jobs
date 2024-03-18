from typing import Optional
from pathlib import Path

from coretex import ComputerVisionDataset, ImageDatasetClasses, ImageDatasetClass, BBox
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


def run(model: YOLO, dataset: ComputerVisionDataset, resultPath: Path, batchSize: int) -> None:
    sampleBatch: list[Path] = []
    for i, sample in enumerate(dataset.samples):
        if any([len(segmentation) < 6 for instance in sample.load().annotation.instances
                for segmentation in instance.segmentations]):
            # Skip invalid annotations (composed of only one or two points)
            continue

        sampleBatch.append(sample.imagePath)
        if (i + 1) % batchSize != 0:
            continue

        results: Results = model.predict(sampleBatch, save = True, project = "./results")
        [processResult(result, dataset.classes, resultPath / f"{sample.id}.png") for result in results]
        sampleBatch.clear()
