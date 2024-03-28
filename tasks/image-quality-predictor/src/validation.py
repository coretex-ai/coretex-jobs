from pathlib import Path

import csv
import logging

from coretex import ComputerVisionSample, folder_manager
from PIL import Image

import torch
import torchvision.transforms as transforms

from .model import CNNModel


def calculateAccuracy(prediction: float, groundtruth: float) -> float:
    accuracy = prediction / groundtruth

    if accuracy > 1:
        accuracy = max(1 - (accuracy - 1), 0)

    return accuracy * 100


def run(modelPath: Path, dataset: list[tuple[ComputerVisionSample, float]], transform: transforms.Compose) -> tuple[Path, float]:
    model = CNNModel()
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    sampleResultsPath = folder_manager.temp / "sample_results.csv"
    sampleAccuracies: list[float] = []

    with sampleResultsPath.open("w") as file:
        writer = csv.DictWriter(file, ["id", "name", "prediction", "groundtruth", "accuracy"])
        writer.writeheader()

        for sample, quality in dataset:
            logging.info(f">> [ImageQuality] Validating sample \"{sample.name}\"...")
            image = Image.open(sample.imagePath).convert("RGB")

            output = model(transform(image).unsqueeze(0)).squeeze(0).item()
            accuracy = calculateAccuracy(output, quality)

            writer.writerow({
                "id": sample.id,
                "name": sample.name,
                "prediction": output,
                "groundtruth": quality,
                "accuracy": f"{accuracy:.2f}"
            })

            sampleAccuracies.append(accuracy)

            logging.info(f"\tGroudtruth: {quality}")
            logging.info(f"\tPrediction: {output}")
            logging.info(f"\tAccuracy: {accuracy}")

    return sampleResultsPath, sum(sampleAccuracies) / len(sampleAccuracies)
