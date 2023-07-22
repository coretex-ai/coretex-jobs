from pathlib import Path

import logging
import pickle
import time

from sklearn.metrics import accuracy_score

from coretex import Experiment, CustomDataset, ExperimentStatus, folder_manager

from .utils import savePredictionFile
from .model import Model
from .dataset import loadDataset, createBatches


def train(experiment: Experiment[CustomDataset], datasetPath: Path, uniqueBodySites: dict[str, int], uniqueTaxons: dict[str, int]) -> float:
    savePredictionFilePath = folder_manager.temp / "body_site_predictions.csv"
    modelPath = folder_manager.temp / "modelFolder"

    experiment.updateStatus(ExperimentStatus.inProgress, "Training LSPIN model")

    validationSplit = experiment.parameters["validationSplit"]
    hiddenLayers = experiment.parameters["hiddenLayers"]
    learningRate = experiment.parameters["learningRate"]
    epochs = experiment.parameters["epochs"]
    displayStep = experiment.parameters["displayStep"]
    batchSize = experiment.parameters["batchSize"]
    bufferSize = experiment.parameters["bufferSize"]
    activationFunc = experiment.parameters["activationFunction"]
    batchNorm = experiment.parameters["batchNorm"]
    seed = experiment.parameters["randomSeed"]
    lZeroLambda = experiment.parameters["lambda"]

    if not (0 <= validationSplit and validationSplit <= 1):
        raise RuntimeError(f">> [MicrobiomeForensics] Validation split must be between 0 and 1. Got {validationSplit}")

    logging.info(">> [MicrobiomeForensics] Instantiating neural network and preparing dataset for training")

    sampleCount = len(list(datasetPath.iterdir()))

    logging.info(f">> [MicrobiomeForensics] Total number of samples is {sampleCount} and {len(uniqueTaxons)} features. {int(sampleCount * (1 - validationSplit))} samples will be used for training, and {sampleCount - int(sampleCount * (1 - validationSplit))} for validation")

    dataset = loadDataset(datasetPath, uniqueBodySites, uniqueTaxons)
    trainData, trainBatches, testData, testBatches = createBatches(
        dataset,
        sampleCount,
        validationSplit,
        bufferSize,
        batchSize
    )

    modelParams = {
        "input_node": len(uniqueTaxons),
        "hidden_layers_node": hiddenLayers,
        "output_node": len(uniqueBodySites),
        "feature_selection": True,
        "batch_normalization": batchNorm,
        "activation_pred": activationFunc,
        "activation_gating": "tanh",
        "gating_net_hidden_layers_node": [10],
        "display_step": displayStep,
        "seed": seed,
        "lam": lZeroLambda
    }

    model = Model(**modelParams)

    logging.info(">> [MicrobiomeForensics] Starting training")

    start = time.time()
    model.train(experiment, trainData, testData, trainBatches, testBatches, epochs, learningRate)
    trainTime = time.time() - start

    yPred, yTest = model.test(testData, testBatches)
    zPred, yTrain = model.test(trainData, trainBatches)

    accuracy = accuracy_score(yTest, yPred)

    logging.info(f">> [MicrobiomeForensics] Training finished in {trainTime:>0.1f}s with accuracy: {accuracy * 100:>0.2f}%")

    trainCount = int((1 - validationSplit) * sampleCount)
    testCount = sampleCount - trainCount

    sampleIds: list[str] = []
    for path in datasetPath.iterdir():
        sampleIds.append(path.name)

    experiment.updateStatus(ExperimentStatus.inProgress, "Saving model and associated data")

    savePredictionFile(
        experiment,
        savePredictionFilePath,
        trainCount,
        testCount,
        sampleIds,
        uniqueBodySites,
        yTrain,
        yTest,
        yPred,
        zPred
    )

    model.save(modelPath / "model")

    with open(modelPath / "uniqueTaxons.pkl", "wb") as f:
        pickle.dump(uniqueTaxons, f)

    with open(modelPath / "uniqueBodySites.pkl", "wb") as f:
        pickle.dump(uniqueBodySites, f)

    return accuracy
