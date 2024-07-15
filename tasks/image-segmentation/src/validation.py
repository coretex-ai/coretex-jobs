from pathlib import Path

import logging
import csv

from coretex import TaskRun, ImageSample, ImageDatasetClasses, Model
from coretex.utils import resizeWithPadding

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils import hasDotAnnotation


def meanIouScore(iouScores: list[float]) -> float:
    return sum(iouScores) / len(iouScores)


def plotImageSegmentation(realImage: np.ndarray, trueSegmentation: np.ndarray, predictedSeqmentation: np.ndarray, iouScore: float, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))

    axes[0].imshow(realImage, cmap = "summer")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(trueSegmentation, cmap = "summer")
    axes[1].set_title("True segmentation")
    axes[1].axis("off")

    axes[2].imshow(predictedSeqmentation, cmap = "summer")
    axes[2].set_title(f"Predicted segmentation\n IoU Score: {iouScore}")
    axes[2].axis("off")

    plt.savefig(f"{name}.jpg")
    plt.close()


def iouScoreClass(testMask: np.ndarray, predictedMask: np.ndarray) -> float:
    intersecrion = np.logical_and(testMask, predictedMask)
    union = np.logical_or(testMask, predictedMask)

    return float(np.sum(intersecrion) / np.sum(union))


def iouScoreImage(testMask: np.ndarray, predictedMask: np.ndarray) -> float:
    iouScores: list[float] = []

    classLabels = np.unique(testMask)
    if np.any(classLabels == 0) and len(classLabels) == 1:
        return 0.0

    for label in classLabels:
        if label > 0:
            testMaskClass = testMask == label
            predictedMaskClass = predictedMask == label
            iouScores.append(iouScoreClass(testMaskClass, predictedMaskClass))

    return sum(iouScores) / len(iouScores)


def iouScoreImages(testMasks: list[np.ndarray], predictedMasks: list[np.ndarray], samples: list[ImageSample]) -> list[float]:
    iouScores: list[float] = []
    for testMask, predictedMask, sample in zip(testMasks, predictedMasks, samples):
        iou = iouScoreImage(testMask, predictedMask)
        iouScores.append(iou)
        logging.info(f">> [Image Segmentation] IoU Score for sample {sample.name} (sample id: {sample.id}) is: {round(iou, 2)}")

    return iouScores


def addPadding(predictedMasks: list[np.ndarray], testMasksPadding: list[tuple[int, int]]) -> list[np.ndarray]:
    for index, mask in enumerate(predictedMasks):
        vPadding = testMasksPadding[index][0]
        hPadding = testMasksPadding[index][1]

        if vPadding > 0:
            mask[:vPadding, :] = 0
            mask[-vPadding:, :] = 0
        if hPadding > 0:
            mask[:, :hPadding] = 0
            mask[:, -hPadding:] = 0

    return predictedMasks


def run(model: tf.lite.Interpreter, batchImages: np.ndarray) -> np.ndarray:
    inputDetails = model.get_input_details()
    outputDetails = model.get_output_details()

    model.resize_tensor_input(inputDetails[0]["index"], batchImages.shape)
    model.allocate_tensors()
    model.set_tensor(inputDetails[0]["index"], batchImages.astype(np.float32))
    model.invoke()

    prediction = model.get_tensor(outputDetails[0]["index"])
    prediction = np.argmax(prediction, axis = -1)

    return prediction


def predictionPrepare(batchSamples: list[ImageSample], imageSize: int, classes: ImageDatasetClasses) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    testImgs: list[np.ndarray] = []
    testMasks: list[np.ndarray] = []
    testMasksPadding: list[tuple[int, int]] = []

    for sample in batchSamples:
        sample.download()
        sample.unzip()
        sampleData = sample.load()

        if hasDotAnnotation(sampleData.annotation):
            logging.warning(f">> [Image Segmentation] Sample \"{sample.name}\" (ID: {sample.id}) has invalid annotation (too few coordinates). Skipping Sample")
            continue

        resized, _, _ = resizeWithPadding(sampleData.image, imageSize, imageSize)
        normalized = resized / 255

        groundtruth = sampleData.extractSegmentationMask(classes)
        groundtruth, vPadding, hPadding = resizeWithPadding(groundtruth, imageSize, imageSize)

        testImgs.append(normalized)
        testMasks.append(groundtruth)
        testMasksPadding.append((vPadding, hPadding))

    testImages = np.array(testImgs)

    return (testImages, testMasks, testMasksPadding)


def loadModel(modelPath: Path) -> tf.lite.Interpreter:
    modelInterpreter = tf.lite.Interpreter(str(modelPath / "model.tflite"))
    return modelInterpreter


def validation(taskRun: TaskRun) -> None:
    dataset = taskRun.dataset
    imageSize: int = taskRun.parameters["imageSize"]
    batchSize: int = taskRun.parameters["batchSize"]
    dataset.classes.exclude(taskRun.parameters["excludedClasses"])

    model: Model = taskRun.parameters["trainedModel"]
    model.download()
    modelInterpreter = loadModel(model.path)

    iouScores: list[float] = []
    fieldNamesSamples = ["Sample ID", "Sample Name", "IoU Score", "Accuracy"]
    csvSamplesData: list[dict[str, str]] = []

    for startIndex in range(0, dataset.count, batchSize):
        batchSamples = dataset.samples[startIndex : startIndex + batchSize]
        testImages, testMasks, testMasksPadding = predictionPrepare(batchSamples, imageSize, dataset.classes)

        prediction = run(modelInterpreter, testImages)
        logging.info(f">> [Image Segmentation] Segmentation prediction for the batch {startIndex // batchSize + 1}/{dataset.count // batchSize + 1} is complete.")
        batchPrediction = [np.array(element) for element in prediction.tolist()]

        predictedMasks = addPadding(batchPrediction, testMasksPadding)

        batchIouScore = iouScoreImages(testMasks, predictedMasks, batchSamples)
        iouScores.extend(batchIouScore)

        for testImg, testSeg, predSeg, iou, sample in zip(testImages, testMasks, predictedMasks, batchIouScore, batchSamples):
            plotImageSegmentation(testImg, testSeg, predSeg, round(iou, 2), str(sample.id))
            logging.info(f">> [Image Segmentation] The results image for sample {sample.name} (sample id: {sample.id}) add to the artifacts.")
            csvSamplesData.append(dict(zip(fieldNamesSamples, [str(sample.id), sample.name, str(round(iou, 2)), str(round(iou * 100))])))

    with open("sample_results.csv", "w", newline = "") as csvFile:
        writer = csv.DictWriter(csvFile, fieldNamesSamples)
        writer.writeheader()
        writer.writerows(csvSamplesData)

    logging.info(f">> [Image Segmentation] The .csv file with sample results has been added to the artifacts.")

    iouScore = meanIouScore(iouScores)
    fieldNamesDataset = ["IoU Score", "IoU STD", "Accuracy"]
    csvDatasetData = dict(zip(fieldNamesDataset, [round(iouScore, 2), round(np.std(iouScores), 2), round(iouScore * 100)]))

    with open("dataset_results.csv", "w", newline = "") as csvFile:
        writer = csv.DictWriter(csvFile, fieldNamesDataset)
        writer.writeheader()
        writer.writerow(csvDatasetData)

    logging.info(f">> [Image Segmentation] The .csv file with dataset results has been added to the artifacts.")
