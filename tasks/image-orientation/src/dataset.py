from typing import Any, Optional
from pathlib import Path

import json

from PIL import Image, ImageOps
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from coretex import ImageDataset, folder_manager

import torch


class OrientedDataset(Dataset):

    def __init__(self, imagesDir: Path, sampleIds: list[int], labelColumn: str, transform: transforms.Compose = None):
        self.imagesDir = imagesDir
        self.sampleIds = sampleIds
        self.labelColumn = labelColumn
        self.transform = transform

    def __len__(self):
        return len(self.sampleIds)

    def __getitem__(self, idx) -> dict[str, Any]:
        imagePath = self.imagesDir / f"{self.sampleIds[idx]}.png"
        metadataPath = self.imagesDir / f"{self.sampleIds[idx]}.json"

        image = ImageOps.exif_transpose(Image.open(imagePath).convert("RGB"))
        with metadataPath.open("r") as file:
            meta = json.load(file)
            flipped = meta.get(self.labelColumn, False)

        label = [1, 0] if flipped else [0, 1]
        label = torch.tensor(label).type(torch.float)

        if self.transform is not None:
            image = self.transform(image)

        sample = {"image": image, "label": label}
        return sample


def getTransform(
    imageSize: tuple[int, int] = (512, 512),
    meanAndStd: Optional[tuple[list[float], list[float]]] = None
) -> transforms.Compose:

    compose = [transforms.Resize(imageSize), transforms.ToTensor()]
    if meanAndStd is not None:
        compose.append(transforms.Normalize(meanAndStd[0], meanAndStd[1]))

    return transforms.Compose(compose)


def prepareDataset(dataset: ImageDataset) -> Path:
    sampleIds: list[int] = []
    imagesDir = folder_manager.createTempFolder("imagesDir")
    for sample in dataset.samples:
        sampleIds.append(sample.id)
        sample.unzip()

        sample.imagePath.link_to(imagesDir / f"{sample.id}.png")
        sample.metadataPath.link_to(imagesDir / f"{sample.id}.json")

    return imagesDir, sampleIds


def splitDataset(dataset: "OrientedDataset", validSplit: float) -> tuple["OrientedDataset", "OrientedDataset"]:
    totalSize = len(dataset)
    trainSize = int((1.0 - validSplit) * totalSize)
    validationSize = totalSize - trainSize

    return random_split(dataset, [trainSize, validationSize])
