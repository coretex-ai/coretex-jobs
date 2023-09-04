import logging
import pickle

from coretex import CustomDataset, Run, Model, MetricType, Metric, folder_manager
from coretex.job import initializeJob

from src.train import train
from src.utils import saveModel
from src.validate import validate
from src.load_data import loadDataAtlas
from src.load_data_std import loadDataStd


def validation(run: Run[CustomDataset]) -> None:
    logging.info(">> [MicrobiomeForensics] Fetching dataset for validation")

    trainedModelId = run.parameters["trainedModel"]

    if trainedModelId is None:
        raise RuntimeError(">> [MicrobiomeForensics] In order to start the validation process You have to type in \"trainedModel\" in run parameters")

    logging.info(f">> [MicrobiomeForensics] Fetching pretrained model from Coretex. Model id: {run.parameters['trainedModel']}")

    datasetPath = folder_manager.createTempFolder("processedDataset")
    trainedModel = Model.fetchById(trainedModelId)
    trainedModel.download()

    # The same dictionaries will be used as during training
    modelPath = folder_manager.modelsFolder / str(run.parameters["trainedModel"])

    with modelPath.joinpath("uniqueBodySites.pkl").open("rb") as f:
        validBodysites = pickle.load(f)

    with modelPath.joinpath("uniqueTaxons.pkl").open("rb") as f:
        validTaxons = pickle.load(f)

    if run.parameters["datasetType"] == 1:
        taxonomicLevel = run.parameters["taxonomicLevel"]
        uniqueBodySites, uniqueTaxons, _ = loadDataStd(
            run,
            run.dataset,
            datasetPath,
            taxonomicLevel,
            validBodysites,
            validTaxons
        )
    else:
        sampleOrigin = run.parameters["sampleOrigin"]
        sequencingTechnique = run.parameters["sequencingTechnique"]
        useCache = run.parameters["cache"]

        uniqueBodySites, uniqueTaxons, _ = loadDataAtlas(
            run,
            run.dataset,
            datasetPath,
            sampleOrigin,
            sequencingTechnique,
            useCache,
            validBodysites,
            validTaxons
        )

    validate(run, datasetPath, uniqueBodySites, uniqueTaxons, trainedModelId)


def training(run: Run[CustomDataset]) -> None:
    epochs = run.parameters["epochs"]

    run.createMetrics([
        Metric.create("train_loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("valid_loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("train_accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1]),
        Metric.create("valid_accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1])
    ])

    folder_manager.createTempFolder("modelFolder")
    datasetPath = folder_manager.createTempFolder("processedDataset")

    if run.parameters["datasetType"] == 1:
        logging.info(">> [MicrobiomeForensics] Standard data selected")

        taxonomicLevel = run.parameters["taxonomicLevel"]

        uniqueBodySites, uniqueTaxons, datasetLen = loadDataStd(run, run.dataset, datasetPath, taxonomicLevel)
        accuracy = train(run, datasetPath, uniqueBodySites, uniqueTaxons)

    else:
        logging.info(">> [MicrobiomeForensics] Raw Microbe Atlas data selected")

        sampleOrigin = run.parameters["sampleOrigin"]
        sequencingTechnique = run.parameters["sequencingTechnique"]
        useCache = run.parameters["cache"]

        uniqueBodySites, uniqueTaxons, datasetLen = loadDataAtlas(
            run,
            run.dataset,
            datasetPath,
            sampleOrigin,
            sequencingTechnique,
            useCache
        )

        accuracy = train(run, datasetPath, uniqueBodySites, uniqueTaxons)

    saveModel(run, accuracy, uniqueBodySites, datasetLen, len(uniqueTaxons))


def main(run: Run[CustomDataset]) -> None:
    validation(run) if run.parameters["validation"] else training(run)


if __name__ == "__main__":
    initializeJob(main)
