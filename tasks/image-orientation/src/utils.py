from typing import Union
from pathlib import Path

from PIL import Image, ImageOps

import torch
import numpy as np


class EarlyStopping:

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._counter = 0

    def __call__(self, bestLoss: Union[float, torch.Tensor], latestLoss: Union[float, torch.Tensor]) -> bool:
        if latestLoss >= bestLoss:
            # Loss did not improve
            self._counter += 1
        else:
            # Loss improved
            self._counter = 0

        return self._counter >= self.patience


def getMeanAndStd(directory: Path) -> tuple[list[float], list[float]]:
    channelsSum, channelsSquaredSum, numImages = 0, 0, 0

    for image_file in directory.glob("*.png"):
        image = np.array(ImageOps.exif_transpose(Image.open(image_file)).convert("RGB"), dtype = np.float32) / 255.0  # Normalize pixel values to [0, 1]

        channelsSum += np.mean(image, axis = (0, 1))
        channelsSquaredSum += np.mean(np.square(image), axis = (0, 1))
        numImages += 1

    mean = channelsSum / numImages
    std = np.sqrt(channelsSquaredSum / numImages - np.square(mean))

    return mean.tolist(), std.tolist()


def calculateAccuracy(yPred: torch.Tensor, yTrue: torch.Tensor) -> float:
    correct = (yPred.argmax(1) == yTrue.argmax(1)).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()
