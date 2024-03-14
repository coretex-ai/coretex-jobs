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
    for sample in dataset.samples:
        result: Results = model.predict(sample.imagePath, save = True, project = "./results")[0]

        fig = plt.figure(num = 1, clear = True)
        plt.imshow(result.orig_img)

        if result.boxes is not None:
            for minX, minY, maxX, maxY, confidence, labelId in result.boxes.data:
                box = BBox.create(minX, minY, maxX, maxY)

                clazz = classByLabelId(int(labelId), dataset.classes)
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

        plt.savefig(resultPath / f"{sample.id}.png")
