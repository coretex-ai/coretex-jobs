import logging
import json

from coretex import currentTaskRun, Artifact, folder_manager, Model, TaskRun, ImageSample
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from src import training, validation
from src.data import ImageQualityDataset, loadDataset, split
from src.model import CNNModel


def fetchArtifacts(validationArtifacts: list[int]) -> list[Artifact]:
    artifacts: list[Artifact] = []

    for taskRunId in validationArtifacts:
        taskRunArtifacts = [artifact for artifact in Artifact.fetchAll(taskRunId) if artifact.remoteFilePath == "sample_results.csv"]

        if len(taskRunArtifacts) == 0:
            logging.error(f">> [ImageQuality] Failed to find artifact \"sample_results.csv\" for Task Run ({taskRunId})")
            continue

        if len(taskRunArtifacts) != 1:
            logging.error(f">> [ImageQuality] Found more than one artifact for Task Run ({taskRunId}) with path \"sample_results.csv\"")
            continue

        artifacts.append(taskRunArtifacts[0])

    return artifacts


def train(taskRun: TaskRun, dataset: list[tuple[ImageSample, float]]) -> None:
    if taskRun.parameters["imageSize"] is None:
        raise RuntimeError("The imageSize for training is not defined")

    imageSize: int = taskRun.parameters["imageSize"]
    if imageSize < 224:
        raise ValueError("Image size cannot be lower than 224")

    # Define transformations for your dataset
    transform = transforms.Compose([
        transforms.Resize((imageSize, imageSize)),
        transforms.ToTensor()
    ])

    if taskRun.parameters["epochs"] is None:
        raise RuntimeError("The number of epochs for training the model is not defined")

    if taskRun.parameters["validationPct"] is None:
        raise RuntimeError("validationSplit parameter is not defined")

    epochs: int = taskRun.parameters["epochs"]
    batchSize: int = taskRun.parameters["batchSize"]

    trainData, validData = split(taskRun.parameters["validationPct"], dataset)

    # Assuming you have your dataset loaded and split into train and test sets
    trainDataset = ImageQualityDataset(trainData, transform)
    trainLoader = DataLoader(trainDataset, batchSize, shuffle = True)

    validDataset = ImageQualityDataset(validData, transform)
    validLoader = DataLoader(validDataset, batchSize)

    # Check if we can train on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logging.info(">> [ImageQuality] Using GPU for training")
    else:
        logging.info(">> [ImageQuality] Using CPU for training")

    # Initialize the model, loss function, and optimizer
    model = CNNModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Move to the target device (GPU or CPU) for training
    model.to(device)
    criterion.to(device)

    modelPath = folder_manager.createTempFolder("model")
    training.run(taskRun, trainLoader, validLoader, model, optimizer, criterion, epochs, modelPath, device)

    Model.saveModelDescriptor(modelPath, {
        "taskRunId": taskRun.id,
        "modelName": taskRun.name,
        "spaceName": taskRun.projectName,
        "projectName": taskRun.taskName,
        "epochs": epochs,
        "batchSize": batchSize,
        "imageSize": taskRun.parameters["imageSize"]
    })

    # Calculate model accuracy
    logging.info(">> [ImageQuality] Validating model...")
    sampleResultsCsvPath, accuracy = validation.run(modelPath / "best.pt", trainData + validData, transform)
    logging.info(f">> [ImageQuality] Model accuracy: {accuracy:.2f}%")

    if taskRun.createArtifact(sampleResultsCsvPath, sampleResultsCsvPath.name) is None:
        logging.error(f">> [ImageQuality] Failed to create artifact \"{sampleResultsCsvPath.name}\"")

    logging.info(">> [ImageQuality] Uploading model...")
    ctxModel = Model.createModel(taskRun.generateEntityName(), taskRun.projectId, accuracy)
    ctxModel.upload(modelPath)

    taskRun.submitOutput("model", ctxModel)


def validate(taskRun: TaskRun, dataset: list[tuple[ImageSample, float]]) -> None:
    if taskRun.parameters["trainedModel"] is None:
        raise RuntimeError("Model id used for image quality prediction that needs validation is not valid")

    modelVal: Model = taskRun.parameters["trainedModel"]
    modelVal.download()
    with open(modelVal.path / modelVal.modelDescriptorFileName(), "r") as file:
        data = json.load(file)

    try:
        imageSize = data["imageSize"]
    except KeyError:
        raise KeyError("The model does not contain information about the imageSize parameter with which it was trained")

    if imageSize < 224:
        raise ValueError("Image size cannot be lower than 224")

    # Define transformations for your dataset
    transform = transforms.Compose([
        transforms.Resize((imageSize, imageSize)),
        transforms.ToTensor()
    ])

    # Calculate model accuracy
    logging.info(">> [ImageQuality] Validating model...")
    sampleResultsCsvPath, datasetResultsCsvPath, accuracy = validation.run(modelVal.path / "best.pt", dataset, transform)
    logging.info(f">> [ImageQuality] Model accuracy: {accuracy:.2f}%")

    if taskRun.createArtifact(sampleResultsCsvPath, sampleResultsCsvPath.name) is None:
        logging.error(f">> [ImageQuality] Failed to create artifact \"{sampleResultsCsvPath.name}\"")

    if taskRun.createArtifact(datasetResultsCsvPath, datasetResultsCsvPath.name) is None:
        logging.error(f">> [ImageQuality] Failed to create artifact \"{sampleResultsCsvPath.name}\"")

def main() -> None:
    taskRun = currentTaskRun()
    artifacts = fetchArtifacts(taskRun.parameters["validationArtifacts"])
    dataset = loadDataset(artifacts)

    if taskRun.parameters["validation"]:
        validate(taskRun, dataset)
    else:
        train(taskRun, dataset)


if __name__ == "__main__":
    main()
