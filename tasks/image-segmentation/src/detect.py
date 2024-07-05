import logging

from keras import Model as KerasModel
from coretex import ImageDataset, TaskRun, folder_manager
from coretex.utils import resizeWithPadding

import numpy as np
import matplotlib.pyplot as plt

from .utils import hasDotAnnotation


def run(taskRun: TaskRun, model: KerasModel, dataset: ImageDataset) -> None:
    imageSize: int = taskRun.parameters["imageSize"]

    for sample in dataset.samples:
        logging.info(f">> [Image Segmentation] Running prediction on sample \"{sample.name}\"")

        sampleData = sample.load()

        annotation = sampleData.annotation
        if annotation is None:
            logging.warning(f">> [Image Segmentation] Sample \"{sample.name}\" (ID: {sample.id}) has no annotation. Skipping Sample")
            continue

        if hasDotAnnotation(annotation):
            logging.warning(f">> [Image Segmentation] Sample \"{sample.name}\" (ID: {sample.id}) has invalid annotation (too few coordinates). Skipping Sample")
            continue

        resized, _, _ = resizeWithPadding(sampleData.image, imageSize, imageSize)
        normalized = resized / 255

        groundtruth = sampleData.extractSegmentationMask(dataset.classes)
        groundtruth, _, _ = resizeWithPadding(groundtruth, imageSize, imageSize)

        prediction = model(np.reshape(normalized, (1,) + normalized.shape), training = False)[0]
        prediction = np.argmax(prediction, axis = -1)

        fig, axes = plt.subplots(1, 3)

        axes[0].set_title("Input image")  # type: ignore[index]
        axes[0].imshow(resized)  # type: ignore[index]

        axes[1].set_title("Groundtruth mask")  # type: ignore[index]
        axes[1].imshow(groundtruth)  # type: ignore[index]

        axes[2].set_title("Predicted mask")  # type: ignore[index]
        axes[2].imshow(prediction)  # type: ignore[index]

        plotPath = folder_manager.temp / f"{sample.id}.png"

        plt.savefig(plotPath)
        plt.close()

        artifact = taskRun.createArtifact(plotPath, plotPath.name)
        if artifact is None:
            logging.warning("\tFailed to upload prediction as artifact")
