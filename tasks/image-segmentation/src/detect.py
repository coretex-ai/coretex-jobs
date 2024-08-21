import logging
import csv

from keras import Model as KerasModel
from coretex import ImageDataset, TaskRun, folder_manager, ImageSample
from coretex.utils import resizeWithPadding

import numpy as np
import matplotlib.pyplot as plt

from .utils import hasDotAnnotation


def iouScoreClass(testMask: np.ndarray, predictedMask: np.ndarray) -> float:
    intersecrion = np.logical_and(testMask, predictedMask)
    union = np.logical_or(testMask, predictedMask)

    return float(np.sum(intersecrion) / np.sum(union) * 100)


def predict(taskRun: TaskRun[ImageDataset], model: KerasModel, samples: list[ImageSample]) -> list[dict[str, str]]:
    imageSize: int = taskRun.parameters["imageSize"]

    batchResult: list[dict[str, str]] = []
    for sample in samples:
        logging.info(f">> [Image Segmentation] Running prediction on sample \"{sample.name}\"")

        sampleData = sample.load()

        if sampleData.annotation is None:
            logging.warning(f">> [Image Segmentation] Sample \"{sample.name}\" (ID: {sample.id}) has no annotation. Skipping Sample")
            continue
        else:
            if hasDotAnnotation(sampleData.annotation):
                logging.warning(f">> [Image Segmentation] Sample \"{sample.name}\" (ID: {sample.id}) has invalid annotation (too few coordinates). Skipping Sample")
                continue

        resized, _, _ = resizeWithPadding(sampleData.image, imageSize, imageSize)
        normalized = resized / 255

        groundtruth = sampleData.extractSegmentationMask(taskRun.dataset.classes)
        groundtruth, _, _ = resizeWithPadding(groundtruth, imageSize, imageSize)

        prediction = model(np.reshape(normalized, (1,) + normalized.shape), training = False)[0]
        prediction = np.argmax(prediction, axis = -1)

        result: dict[str, str] = {}
        result["sample id"] = f"{sample.id}"
        result["sample name"] = sample.name
        for classId, className in enumerate(taskRun.dataset.classes.labels):
            groundtruthClassMask = np.zeros_like(groundtruth)
            groundtruthClassMask[groundtruth == classId + 1] = classId + 1

            predictionClassMask = np.zeros_like(prediction)
            predictionClassMask[prediction == classId + 1] = classId + 1

            iouClass = iouScoreClass(groundtruthClassMask, predictionClassMask)
            result[className] = f"{iouClass:.2f}"


        iou = iouScoreClass(groundtruth, prediction)
        result["accuracy"] = f"{iou:.2f}"
        batchResult.append(result)

        fig, axes = plt.subplots(1, 3, figsize = (15, 5))

        axes[0].set_title(f"Input image, id: {sample.id}")
        axes[0].imshow(resized, cmap = "summer")
        axes[0].axis("off")

        axes[1].set_title("Groundtruth mask")
        axes[1].imshow(groundtruth, cmap = "summer")
        axes[1].axis("off")

        axes[2].set_title(f"Predicted mask\nAcc: {iou:.2f}")
        axes[2].imshow(prediction, cmap = "summer")
        axes[2].axis("off")

        plotPath = folder_manager.temp / f"{sample.id}.png"
        plt.savefig(plotPath)
        plt.close()

        if taskRun.createArtifact(plotPath, f"sample_results/{sample.id}/{plotPath.name}") is None:
            logging.warning(f">> [Image Segmentation] Failed to upload image \"{plotPath.name}\" with segmentation as artifact")
        else:
            logging.info(f">> [Image Segmentation] The segmentation image \"{plotPath.name}\" has been uploaded as artifact")

        csvSamplePath = folder_manager.temp.joinpath(f"{sample.id}.csv")
        with csvSamplePath.open("w") as file:
            writer = csv.DictWriter(file, fieldnames = result.keys())
            writer.writeheader()
            writer.writerow(result)

        if taskRun.createArtifact(csvSamplePath, f"sample_results/{sample.id}/{sample.id}.csv") is None:
            logging.warning(f">> [Image Segmentation] Failed to upload csv file with results for sample {sample.id} as artifact")
        else:
            logging.info(f">> [Image Segmentation] The csv file with results for sample {sample.id} has been uploaded as artifact")

    return batchResult
