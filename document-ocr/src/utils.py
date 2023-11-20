from pathlib import Path

import logging

from PIL import Image, ImageDraw, ImageFont

import numpy as np
import matplotlib.pyplot as plt

from coretex import folder_manager, TaskRun, ImageSample, ImageDataset, BBox


def createArtifact(taskRun: TaskRun[ImageDataset], filePath: Path, artifactPath: str) -> None:
    if taskRun.createArtifact(filePath, artifactPath) is None:
        logging.error(f"\tFailed to create artifact \"{artifactPath}\"")


def savePlot(sample: ImageSample, predictedMask: np.ndarray, taskRun: TaskRun) -> None:
    plotPath = folder_manager.temp / f"segmentation.png"
    fig, axes = plt.subplots(1, 3)

    originalImage = Image.open(sample.imagePath)

    sampleData = sample.load()

    groundTruth = None
    for instance in sampleData.annotation.instances:
        if taskRun.dataset.classes.classById(instance.classId).label == "document":
            groundTruth = instance.extractBinaryMask(originalImage.size[0], originalImage.size[1])
            continue

    if groundTruth is None:
        logging.warning(">> [Document OCR] No ground truth mask found")
        return

    axes[0].set_title("Input image")
    axes[0].imshow(originalImage)

    axes[1].set_title("Groundtruth mask")
    axes[1].imshow(groundTruth)

    axes[2].set_title("Predicted mask")
    axes[2].imshow(predictedMask)

    plt.savefig(plotPath)
    plt.close()

    createArtifact(taskRun, plotPath, f"{sample.name}/segmentation.png")
    plotPath.unlink()


def saveDocumentWithDetections(image: Image.Image, bboxes: list[BBox], classes: list[str], outputDir: Path, taskRun: TaskRun):
    font = ImageFont.load_default()

    for i, bbox in enumerate(bboxes):
        xy = [(0, 0), (bbox.width, 0), (bbox.width, bbox.height), (0, bbox.height)]
        color = (255, 255, 255, 92)
        outline = (255, 255, 255, 212)
        polyOffset = (bbox.minX, bbox.minY)

        polygon = Image.new("RGBA", (bbox.width,  bbox.height))
        pdraw = ImageDraw.Draw(polygon)
        pdraw.polygon(xy, color, outline, width = 2)
        image.paste(polygon, polyOffset, mask = polygon)

        textOffset = (bbox.minX, bbox.minY - 20)

        text = Image.new("RGBA", (image.width, 20))
        tdraw = ImageDraw.Draw(text)
        textSize = tdraw.textbbox((0, 0), classes[i], font = font)
        tdraw.rectangle(textSize, color)
        tdraw.text((0, 0), classes[i], fill="white", font = font)
        image.paste(text, textOffset, mask = text)


    savePath = outputDir / "detections.png"
    image.save(savePath)

    createArtifact(taskRun, savePath, savePath.relative_to(outputDir.parent))
