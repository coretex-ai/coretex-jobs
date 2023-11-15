import logging

from coretex import currentTaskRun, ImageDataset, ImageSample, Model, folder_manager

from src import detect
from src.model import loadModel
from src.image_segmentation import processMask, generateSegmentedImage
from src.utils import savePlot


def main() -> None:
    taskRun = currentTaskRun()

    outputdir = folder_manager.createTempFolder("images")

    dataset: ImageDataset = taskRun.dataset
    dataset.download()

    coretexModel: Model = taskRun.parameters["model"]
    coretexModel.download()

    model = loadModel(coretexModel.path / "model.tflite")
    predictions = detect.run(model, dataset)

    outputDataset = ImageDataset.createDataset(f"{taskRun.id}-segmented-images", taskRun.projectId)
    if outputDataset is None:
        raise ValueError(">> [Document OCR] Failed to create output dataset")

    for i, sample in enumerate(dataset.samples):
        logging.info(f">> [Document OCR] Performig segmentation on sample \"{sample.name}\"")

        mask = processMask(predictions[i])

        image = generateSegmentedImage(sample.imagePath, mask)
        if image is None:
            continue

        savePlot(sample.name, image, mask, taskRun)

        savePath = outputdir / f"{sample.name}.png"
        image.save(savePath)
        ImageSample.createImageSample(outputDataset.id, savePath)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
