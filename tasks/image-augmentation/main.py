import logging

from coretex import TaskRun, ImageDataset, currentTaskRun

import albumentations as A

from src.augmentation import augmentImage
from src.utils import copySample, getRelevantParameters


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    dependencies = getRelevantParameters(taskRun)

    # try:
    #     cache = ImageDataset.fetchCachedDataset(dependencies)
    #     taskRun.submitOutput("outputDataset", cache)
    # except ValueError as e:
    #     pass

    dataset = taskRun.dataset
    dataset.download()

    outputDataset = ImageDataset.createCacheDataset(f"{taskRun.id}-imgaug", dependencies, taskRun.projectId)
    outputDataset.saveClasses(dataset.classes)

    flipH = taskRun.parameters["flipHorizontalPct"]
    flipV = taskRun.parameters["flipVerticalPct"]
    rotate = taskRun.parameters["rotate"]
    rotate180 = taskRun.parameters["rotate180Pct"]
    noise = taskRun.parameters["noise"]
    blurPct = taskRun.parameters["blurPct"]
    blurLimit = taskRun.parameters["blurLimit"]
    brightness = taskRun.parameters["brightness"]
    contrast = taskRun.parameters["contrast"]
    keepOriginalImages = taskRun.parameters["keepOriginalImages"]

    augmentersGeometric: list[A.BasicTransform] = []
    augmentersPhotometric: list[A.BasicTransform] = []

    if flipH is not None:
        augmentersGeometric.append(A.HorizontalFlip(p = flipH))

    if flipV is not None:
        augmentersGeometric.append(A.VerticalFlip(p = flipV))

    if rotate is not None:
        augmentersGeometric.append(A.Rotate(rotate))

    if rotate180:
        augmentersGeometric.append(A.Rotate((180, 180), p = rotate180))

    if noise is not None:
        augmentersPhotometric.append(A.GaussNoise(noise))

    if blurPct is not None:
        augmentersPhotometric.append(A.Blur(blurLimit, p = blurPct))

    if contrast is not None or brightness is not None:
        augmentersPhotometric.append(A.RandomBrightnessContrast(brightness, contrast))

    transformPhotometric = A.ReplayCompose(augmentersPhotometric)

    for index, sample in enumerate(dataset.samples):
        logging.info(f">> [Image Augmentation] Augmenting Sample \"{sample.name}\" - {index + 1}/{dataset.count}...")

        augmentImage(
            augmentersGeometric,
            transformPhotometric,
            sample,
            taskRun,
            outputDataset
        )

        if keepOriginalImages:
            logging.info("\tCopying original image...")
            copySample(sample, outputDataset)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
