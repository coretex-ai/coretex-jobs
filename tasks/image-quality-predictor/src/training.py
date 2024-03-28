from typing import Optional
from pathlib import Path

import logging

from coretex import TaskRun, Metric, MetricType
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

from .model import CNNModel
from .utils import EarlyStopping


def trainEpoch(trainLoader: DataLoader, model: CNNModel, optimizer: optim.Adam, criterion: nn.MSELoss) -> torch.Tensor:
    model.train()

    runningLoss = torch.tensor(0, dtype = torch.float)

    for batch, (images, labels) in enumerate(trainLoader):
        optimizer.zero_grad()
        outputs = model(images)
        batchLoss = criterion(outputs, labels.float())
        batchLoss.backward()
        optimizer.step()

        runningLoss += batchLoss

    return runningLoss / len(trainLoader)


def computeValLoss(validLoader: DataLoader, model: CNNModel, criterion: nn.MSELoss) -> torch.Tensor:
    model.eval()

    with torch.no_grad():
        totalValidLoss = torch.tensor(0, dtype = torch.float)

        for images, labels in validLoader:
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            totalValidLoss += loss

        return totalValidLoss / len(validLoader)


def run(
    taskRun: TaskRun,
    trainLoader: DataLoader,
    validLoader: DataLoader,
    model: CNNModel,
    optimizer: optim.Adam,
    criterion: nn.MSELoss,
    epochs: int,
    modelPath: Path
) -> None:

    taskRun.createMetrics([
        Metric.create("trainLoss", "epoch", MetricType.int, "loss", MetricType.float, [1, epochs]),
        Metric.create("validLoss", "epoch", MetricType.int, "loss", MetricType.float, [1, epochs])
    ])

    earlyStopping = EarlyStopping(max(10, int(epochs * 0.1)))
    bestLoss: Optional[torch.Tensor] = None

    # Training loop
    for epoch in range(epochs):
        logging.info(f">> [ImageQuality] Started epoch {epoch + 1}/{epochs}")

        trainLoss = trainEpoch(trainLoader, model, optimizer, criterion)
        validLoss = computeValLoss(validLoader, model, criterion)

        taskRun.submitMetrics({
            "trainLoss": (epoch + 1, trainLoss.item()),
            "validLoss": (epoch + 1, validLoss.item())
        })

        if bestLoss is None:
            bestLoss = validLoss

        if earlyStopping(bestLoss, validLoss):
            logging.info(f">> [ImageQuality] Loss did not improve for {earlyStopping.patience} epochs, stopping the training...")
            break

        if bestLoss > validLoss:
            bestLoss = validLoss

            # Save the best model
            torch.save(model.state_dict(), modelPath / "best.pt")

        # Save the latest model
        torch.save(model.state_dict(), modelPath / "last.pt")

        logging.info(f">> [ImageQuality] Finished epoch {epoch + 1}/{epochs}, Train loss: {trainLoss}, valid loss: {validLoss}")
