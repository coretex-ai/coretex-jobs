from typing import Optional
from pathlib import Path

from coretex import ComputerVisionDataset, ImageDatasetClasses, ImageDatasetClass, BBox
from ultralytics import YOLO
from ultralytics.engine.results import Results

import matplotlib.pyplot as plt
import matplotlib.patches as pth


def classByLabelId(labelId: int, classes: ImageDatasetClasses) -> Optional[ImageDatasetClass]:
    return classes.classByLabel(classes.labels[labelId])


def run(model: YOLO, dataset: ComputerVisionDataset, resultPath: Path) -> None:
    results: list[Results] = model.predict(
        [sample.imagePath for sample in dataset.samples],
        save = True,
        project = "./results"
    )

    for sample, result in zip(dataset.samples, results):
        plt.imshow(result.orig_img)

        if result.boxes is not None:
            for minX, minY, maxX, maxY, confidence, labelId in result.boxes.data:
                box = BBox.create(minX, minY, maxX, maxY)

                clazz = classByLabelId(int(labelId), dataset.classes)
                if clazz is None:
                    continue

                plt.gca().add_patch(pth.Rectangle(
                    (box.minX, box.minY),
                    box.width,
                    box.height,
                    linewidth = 3,
                    edgecolor = clazz.color,
                    facecolor = "none"
                ))

        plt.savefig(resultPath / f"{sample.id}.png")
        plt.close()
