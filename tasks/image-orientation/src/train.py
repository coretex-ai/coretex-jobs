from typing import Optional
from pathlib import Path

import logging

from coretex import TaskRun
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

from .model import OrientationClassifier
from .utils import calculateAccuracy, EarlyStopping


def trainEpoch(
    trainLoader: DataLoader,
    model: OrientationClassifier,
    optimizer: optim.Adam,
    criterion: nn.CrossEntropyLoss,
    device: torch.device
) -> tuple[float, float]:

    model.train()

    runningTrainLoss = 0.0
    runningTrainAcc = 0.0
    for data in trainLoader:
        inputs = data["image"].to(device)
        labels = data["label"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        runningTrainLoss += loss.item()
        runningTrainAcc += calculateAccuracy(outputs, labels)

    trainingLoss = runningTrainLoss / len(trainLoader)
    trainingAccuracy = runningTrainAcc / len(trainLoader)

    return trainingLoss, trainingAccuracy


def computeValData(
    validLoader: DataLoader,
    model: OrientationClassifier,
    criterion: nn.CrossEntropyLoss,
    device: torch.device
) -> tuple[float, float]:

    model.eval()

    runningValLoss = 0.0
    runningValAcc = 0.0

    with torch.no_grad():
        for data in validLoader:
            inputs = data["image"].to(device)
            labels = data["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            runningValLoss += loss.item()
            runningValAcc += calculateAccuracy(outputs, labels)

    validationLoss = runningValLoss / len(validLoader)
    validationAccuracy = runningValAcc / len(validLoader)

    return validationLoss, validationAccuracy


def runTraining(
    model: OrientationClassifier,
    trainLoader: DataLoader,
    validLoader: DataLoader,
    optimizer: optim.Adam,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epochs: int,
    taskRun: TaskRun,
    modelPath: Path,
    imageSize: int
) -> None:

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.3, patience = max(5, int(epochs * 0.05)))
    earlyStopping = EarlyStopping(max(10, int(epochs * 0.1)))
    bestLoss: Optional[float] = None
    exampleInput = torch.randn(1, 3, imageSize, imageSize)

    for epoch in range(epochs):
        trainingLoss, trainingAccuracy = trainEpoch(trainLoader, model, optimizer, criterion,  device)
        validationLoss, validationAccuracy = computeValData(validLoader, model, criterion, device)
        scheduler.step(validationLoss)

        logging.info(f">> [Orientation] Epoch: {epoch + 1} || t_loss: {trainingLoss:.4f} | v_loss: {validationLoss:.4f} || t_acc: {trainingAccuracy:.4f} | v_acc: {validationAccuracy:.4f} || LR: {scheduler.get_last_lr()}")
        taskRun.submitMetrics({
            "training_loss": (epoch + 1, trainingLoss),
            "validation_loss": (epoch + 1, validationLoss),
            "training_accuracy": (epoch + 1, trainingAccuracy),
            "validation_accuracy": (epoch + 1, validationAccuracy)
        })

        if bestLoss is None:
            bestLoss = validationLoss

        if earlyStopping(bestLoss, validationLoss):
            logging.info(f">> [Orientation] Loss did not improve for {earlyStopping.patience} epochs, stopping the training...")
            break

        if bestLoss > validationLoss:
            bestLoss = validationLoss

            # Save the best model
            tsModel = torch.jit.trace(model, exampleInput)  # type: ignore[no-untyped-call]
            tsModel.save(modelPath / "best.pt")

        # Save the latest model
        tsModel = torch.jit.trace(model, exampleInput)  # type: ignore[no-untyped-call]
        tsModel.save(modelPath / "last.pt")

        if not modelPath.joinpath("best.pt").exists():
            tsModel = torch.jit.trace(model, exampleInput)  # type: ignore[no-untyped-call]
            tsModel.save(modelPath / "best.pt")
