from typing import Optional, Any

import random
import logging
import csv

from PIL import Image
from coretex import ComputerVisionSample, Artifact
from torch.utils.data import Dataset

import torchvision.transforms as transforms


def isValidationSplitValid(validationPct: float, datasetSize: int) -> bool:
    if not 0 <= validationPct < 1:
        logging.error(f">> [ObjectDetection] validationSplit parameter ({validationPct}) must be between 0 and 1")
        return False

    minSamplesForSplit = int(1 / min(validationPct, 1 - validationPct))
    if datasetSize < minSamplesForSplit:
        logging.error(
            f">> [ObjectDetection] Dataset is too small ({datasetSize}) for validationSplit parameter ({validationPct}). "
            f"Minimum number of samples is {minSamplesForSplit}"
        )

        return False

    return True


def split(
    validationPct: float,
    dataset: list[tuple[ComputerVisionSample, float]]
) -> tuple[list[tuple[ComputerVisionSample, float]], list[tuple[ComputerVisionSample, float]]]:

    if not isValidationSplitValid(validationPct, len(dataset)):
        raise ValueError("Invalid \"validationPct\" value")

    # Shuffle the dataset in predictable manner
    random.seed(42)
    random.shuffle(dataset)

    validCount = int(len(dataset) * validationPct)
    trainCount = len(dataset) - validCount

    return dataset[:trainCount], dataset[trainCount:]


def loadDataset(artifact: Artifact) -> list[tuple[ComputerVisionSample, float]]:
    logging.info(">> [ImageQuality] Loading dataset...")

    artifact.localFilePath.parent.mkdir(exist_ok = True)
    if not artifact.download():
        raise RuntimeError(f"Failed to download artifact {artifact.taskRunId} - {artifact.remoteFilePath}")

    with artifact.localFilePath.open("r") as file:
        dataset: list[tuple[ComputerVisionSample, float]] = []

        for row in csv.DictReader(file):
            sampleId = int(row["id"])
            logging.info(f"\tLoading sample {sampleId}...")

            sample = ComputerVisionSample.fetchById(sampleId)
            sample.download()
            sample.unzip()

            value = (sample, float(row["total_iou"]))
            dataset.append(value)

        return dataset


class ImageQualityDataset(Dataset):

    def __init__(
        self,
        data: list[tuple[ComputerVisionSample, float]],
        transform: Optional[transforms.Compose] = None
    ) -> None:

        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Any, float]:
        sample, quality = self.data[idx]

        image = Image.open(sample.imagePath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, quality
