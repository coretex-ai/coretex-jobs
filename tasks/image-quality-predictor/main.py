import logging

from coretex import currentTaskRun, Artifact, folder_manager, Model
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from src import training, validation
from src.data import ImageQualityDataset, loadDataset, split
from src.model import CNNModel


def main() -> None:
    taskRun = currentTaskRun()
    artifacts = [artifact for artifact in Artifact.fetchAll(taskRun.parameters["validationArtifacts"]) if artifact.remoteFilePath == "sample_results.csv"]

    if len(artifacts) == 0:
        raise RuntimeError("Failed to find artifact \"sample_results.csv\"")

    if len(artifacts) != 1:
        raise RuntimeError("Found more than one artifact with path \"sample_results.csv\"")

    epochs: int = taskRun.parameters["epochs"]
    batchSize: int = taskRun.parameters["batchSize"]

    imageSize: int = taskRun.parameters["imageSize"]
    if imageSize < 224:
        raise ValueError("Image size cannot be lower than 224")

    dataset = loadDataset(artifacts[0])
    trainData, validData = split(taskRun.parameters["validationPct"], dataset)

    # Define transformations for your dataset
    transform = transforms.Compose([
        transforms.Resize((imageSize, imageSize)),
        transforms.ToTensor()
    ])

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
        "imageSize": imageSize
    })

    # Calculate model accuracy
    logging.info(">> [ImageQuality] Validating model...")
    sampleResultsCsvPath, accuracy = validation.run(modelPath / "best.pt", trainData + validData, transform)
    logging.info(f">> [ImageQuality] Model accuracy: {accuracy:.2f}%")

    if taskRun.createArtifact(sampleResultsCsvPath, sampleResultsCsvPath.name) is None:
        logging.error(f">> [ImageQuality] Failed to create artifact \"{sampleResultsCsvPath.name}\"")

    logging.info(">> [ImageQuality] Uploading model...")
    ctxModel = Model.createModel(taskRun.generateEntityName(), taskRun.id, accuracy)
    ctxModel.upload(modelPath)

    taskRun.submitOutput("model", ctxModel)


if __name__ == "__main__":
    main()
