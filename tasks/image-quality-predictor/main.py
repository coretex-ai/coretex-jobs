import logging

from coretex import currentTaskRun, Artifact, folder_manager, Model
from torch.utils.data import DataLoader

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

    # Initialize the model, loss function, and optimizer
    model = CNNModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    modelPath = folder_manager.createTempFolder("model")
    training.run(taskRun, trainLoader, validLoader, model, optimizer, criterion, epochs, modelPath)

    # Calculate model accuracy
    logging.info(">> [ImageQuality] Validating model...")
    sampleResultsCsvPath, accuracy = validation.run(modelPath / "best.pt", trainData + validData, transform)
    logging.info(f">> [ImageQuality] Model accuracy: {accuracy:.2f}%")

    if taskRun.createArtifact(sampleResultsCsvPath, sampleResultsCsvPath.name) is None:
        logging.error(f">> [ImageQuality] Failed to create artifact \"{sampleResultsCsvPath.name}\"")

    logging.info(">> [ImageQuality] Uploading model...")
    ctxModel = Model.createModel(taskRun.name, taskRun.id, 0.5)
    ctxModel.upload(modelPath)

    taskRun.submitOutput("model", ctxModel)


if __name__ == "__main__":
    main()
