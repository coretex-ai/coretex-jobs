import shutil
import logging

from coretex import currentTaskRun, ImageDataset, folder_manager

from src import detect_document
from src.model import loadSegmentationModel, getObjectDetectionModel
from src.image_segmentation import processMask, segmentImage, segmentDetections
from src.ocr import performOCR
from src.utils import savePlot, saveDocumentWithDetections
from src.object_detection import runObjectDetection


def main() -> None:
    taskRun = currentTaskRun()

    outputDir = folder_manager.createTempFolder("sampleOutputs")

    dataset: ImageDataset = taskRun.dataset
    dataset.download()

    segmentationModel = loadSegmentationModel(taskRun.parameters["segmentationModel"])
    predictedMasks = detect_document.run(segmentationModel, dataset)

    objetDetectionModel = getObjectDetectionModel(taskRun.parameters["objectDetectionModel"])

    for i, sample in enumerate(dataset.samples):
        logging.info(f">> [Document OCR] Performing segmentation on sample \"{sample.name}\"")
        sampleOutputdir = outputDir / f"{sample.name}"
        sampleOutputdir.mkdir()

        mask = processMask(predictedMasks[i])

        segmentedImage = segmentImage(sample.imagePath, mask)
        if segmentedImage is None:
            continue

        savePlot(sample, mask, taskRun)

        bboxes, classes = runObjectDetection(segmentedImage, objetDetectionModel)
        segmentedDetections = segmentDetections(segmentedImage, bboxes, classes, sampleOutputdir, taskRun)

        saveDocumentWithDetections(segmentedImage, bboxes, classes, sampleOutputdir, taskRun)

        performOCR(segmentedDetections, classes, sampleOutputdir, taskRun)

        shutil.rmtree(sampleOutputdir)


if __name__ == "__main__":
    main()
