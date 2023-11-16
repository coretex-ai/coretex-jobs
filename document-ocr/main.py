import logging

from coretex import currentTaskRun, ImageDataset, folder_manager

from src import detect_document
from src.model import loadSegmentationModel, getWeights
from src.image_segmentation import processMask, segmentImage, segmentDetections
from src.ocr import performOCR
from src.utils import savePlot, saveDocumentWithDetections, removeDuplicalteDetections
from src.object_detection.detect import run as runObjectDetection


def main() -> None:
    taskRun = currentTaskRun()

    outputDir = folder_manager.createTempFolder("sampleOutputs")

    dataset: ImageDataset = taskRun.dataset
    dataset.download()

    segmentationModel = loadSegmentationModel(taskRun.parameters["segmentationModel"])
    predictedMasks = detect_document.run(segmentationModel, dataset)

    objDetModelWeights = getWeights(taskRun.parameters["objectDetectionModel"])

    for i, sample in enumerate(dataset.samples):
        logging.info(f">> [Document OCR] Performing segmentation on sample \"{sample.name}\"")
        sampleOutputdir = outputDir / f"{sample.name}"
        sampleOutputdir.mkdir()

        mask = processMask(predictedMasks[i])

        segmentedImage = segmentImage(sample.imagePath, mask)
        if segmentedImage is None:
            continue

        segmentedOutput = sampleOutputdir / "segmented.png"
        segmentedImage.save(segmentedOutput)

        savePlot(sample, mask, taskRun)

        bboxes, classes = runObjectDetection(segmentedOutput, objDetModelWeights)
        bboxes, classes = removeDuplicalteDetections(bboxes, classes)

        segmentedDetections, labels = segmentDetections(segmentedImage, bboxes, classes, sampleOutputdir, taskRun)

        saveDocumentWithDetections(segmentedImage, bboxes, classes, sampleOutputdir, taskRun)

        performOCR(segmentedDetections, labels, sampleOutputdir, taskRun)


if __name__ == "__main__":
    main()
