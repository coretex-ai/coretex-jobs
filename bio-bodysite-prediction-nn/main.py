import logging
import pickle

from coretex import CustomDataset, Experiment, Model, MetricType, Metric, folder_manager
from coretex.project import initializeProject

from src.train import train
from src.utils import saveModel
from src.validate import validate
from src.load_data import loadDataAtlas
from src.load_data_std import loadDataStd


def validation(experiment: Experiment[CustomDataset]) -> None:
    logging.info(">> [MicrobiomeForensics] Fetching dataset for validation")

    trainedModelId = experiment.parameters["trainedModel"]

    if trainedModelId is None:
        raise RuntimeError(">> [MicrobiomeForensics] In order to start the validation process You have to type in \"trainedModel\" in experiment parameters")

    logging.info(f">> [MicrobiomeForensics] Fetching pretrained model from Coretex. Model id: {experiment.parameters['trainedModel']}")

    datasetPath = folder_manager.createTempFolder("processedDataset")
    trainedModel = Model.fetchById(trainedModelId)
    trainedModel.download()

    # The same dictionaries will be used as during training
    modelPath = folder_manager.modelsFolder / str(experiment.parameters["trainedModel"])

    with modelPath.joinpath("uniqueBodySites.pkl").open("rb") as f:
        validBodysites = pickle.load(f)

    with modelPath.joinpath("uniqueTaxons.pkl").open("rb") as f:
        validTaxons = pickle.load(f)

    if experiment.parameters["datasetType"] == 1:
        taxonomicLevel = experiment.parameters["taxonomicLevel"]
        uniqueBodySites, uniqueTaxons, _ = loadDataStd(
            experiment,
            experiment.dataset,
            datasetPath,
            taxonomicLevel,
            validBodysites,
            validTaxons
        )
    else:
        sampleOrigin = experiment.parameters["sampleOrigin"]
        sequencingTechnique = experiment.parameters["sequencingTechnique"]
        useCache = experiment.parameters["cache"]

        uniqueBodySites, uniqueTaxons, _ = loadDataAtlas(
            experiment,
            experiment.dataset,
            datasetPath,
            sampleOrigin,
            sequencingTechnique,
            useCache,
            validBodysites,
            validTaxons
        )

    validate(experiment, datasetPath, uniqueBodySites, uniqueTaxons, trainedModelId)


def training(experiment: Experiment[CustomDataset]) -> None:
    epochs = experiment.parameters["epochs"]

    experiment.createMetrics([
        Metric.create("train_loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("valid_loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("train_accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1]),
        Metric.create("valid_accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1])
    ])

    folder_manager.createTempFolder("modelFolder")
    datasetPath = folder_manager.createTempFolder("processedDataset")

    if experiment.parameters["datasetType"] == 1:
        logging.info(">> [MicrobiomeForensics] Standard data selected")

        taxonomicLevel = experiment.parameters["taxonomicLevel"]

        uniqueBodySites, uniqueTaxons, datasetLen = loadDataStd(experiment, experiment.dataset, datasetPath, taxonomicLevel)
        accuracy = train(experiment, datasetPath, uniqueBodySites, uniqueTaxons)

    else:
        logging.info(">> [MicrobiomeForensics] Raw Microbe Atlas data selected")

        sampleOrigin = experiment.parameters["sampleOrigin"]
        sequencingTechnique = experiment.parameters["sequencingTechnique"]
        useCache = experiment.parameters["cache"]

        uniqueBodySites, uniqueTaxons, datasetLen = loadDataAtlas(
            experiment,
            experiment.dataset,
            datasetPath,
            sampleOrigin,
            sequencingTechnique,
            useCache
        )

        accuracy = train(experiment, datasetPath, uniqueBodySites, uniqueTaxons)

    saveModel(experiment, accuracy, uniqueBodySites, datasetLen, len(uniqueTaxons))


def main(experiment: Experiment[CustomDataset]) -> None:
    validation(experiment) if experiment.parameters["validation"] else training(experiment)


if __name__ == "__main__":
    initializeProject(main)
