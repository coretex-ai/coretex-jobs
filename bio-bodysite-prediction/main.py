import logging

from coretex import CustomDataset, Run, Model, MetricType, Metric, folder_manager
from coretex.job import initializeJob

from src.train import train
from src.utils import saveModel
from src.validate import validate
from src.load_data import loadDataAtlas
from src.load_data_std import loadDataStd, prepareForTrainingStd


def validation(run: Run[CustomDataset]) -> None:
    trainedModelId = run.parameters["trainedModel"]

    if trainedModelId is None:
        raise RuntimeError(">> [MicrobiomeForensics] In order to start the validation process You have to type in \"trainedModel\" in run parameters")

    logging.info(f">> [MicrobiomeForensics] Fetching pretrained model from Coretex. Model id: {run.parameters['trainedModel']}")

    trainedModel = Model.fetchById(trainedModelId)
    trainedModel.download()

    inputMatrix, outputMatrix, sampleIdList, uniqueBodySites, _ = loadDataAtlas(run.dataset, run)

    validate(run, inputMatrix, outputMatrix, uniqueBodySites, sampleIdList, trainedModelId)


def training(run: Run[CustomDataset]) -> None:
    epochs = run.parameters["epochs"]

    run.createMetrics([
        Metric.create("loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1])
    ])

    folder_manager.createTempFolder("modelFolder")

    if run.parameters["datasetType"] == 1:
        logging.info(">> [MicrobiomeForensics] Standard data selected")

        level, datasetLen, uniqueTaxons, uniqueBodySites = loadDataStd(run.dataset, run)
        inputMatrix, outputMatrix, sampleIdList = prepareForTrainingStd(level, datasetLen, uniqueTaxons, uniqueBodySites, run)

        accuracy = train(run, inputMatrix, outputMatrix, uniqueBodySites, uniqueTaxons, sampleIdList)

    else:
        logging.info(">> [MicrobiomeForensics] Raw Microbe Atlas data selected")

        inputMatrix, outputMatrix, sampleIdList, uniqueBodySites, uniqueTaxons = loadDataAtlas(run.dataset, run)
        datasetLen = inputMatrix.shape[0]

        accuracy = train(run, inputMatrix, outputMatrix, uniqueBodySites, uniqueTaxons, sampleIdList)

    saveModel(accuracy, uniqueBodySites, datasetLen, len(uniqueTaxons), run)


def main(run: Run[CustomDataset]) -> None:
    validation(run) if run.parameters["validation"] else training(run)


if __name__ == "__main__":
    initializeJob(main)
