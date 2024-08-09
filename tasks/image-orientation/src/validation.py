from pathlib import Path

from coretex import folder_manager, TaskRun
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import torch
import seaborn as sns
import matplotlib.pyplot as plt

from .model import OrientationClassifier
from .utils import calculateAccuracy


def validate(
    dataLoader: DataLoader,
    model: OrientationClassifier,
    device: torch.device,
    confMatrixPath: Path
) -> float:

    predictedOrientations: list[int] = []
    trueOrientations: list[int] = []

    with torch.no_grad():
        for data in dataLoader:
            inputs = data["image"].to(device)
            labels = data["label"].to(device)
            outputs = model(inputs)

            acccuracy = calculateAccuracy(outputs, labels)

            predictions = outputs >= 0.5
            predictedOrientations.extend(predictions[:, 0].tolist())
            trueOrientations.extend(labels[:, 0].tolist())

    cm = confusion_matrix(predictedOrientations, trueOrientations)

    # Plot confusion matrix
    plot = sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
    plt.xlabel("Predicted flipped")
    plt.ylabel("True flipped")
    plt.title("Confusion Matrix")

    plot.get_figure().savefig(confMatrixPath)
    plt.close()

    return acccuracy


def runValidation(
    modelPath: Path,
    trainLoader: DataLoader,
    validLoader: DataLoader,
    device: torch.device,
    taskRun: TaskRun
) -> float:

    model = torch.jit.load(modelPath)  # type: ignore[no-untyped-call]
    model.to(device)
    model.eval()

    valMatrixPath = folder_manager.temp / "confusion_matrix_val.png"
    acccuracy = validate(validLoader, model, device, valMatrixPath)
    taskRun.createArtifact(valMatrixPath, valMatrixPath.name)

    trainMatrixPath = folder_manager.temp / "confusion_matrix_train.png"
    validate(trainLoader, model, device, trainMatrixPath)
    taskRun.createArtifact(trainMatrixPath, trainMatrixPath.name)

    return acccuracy
