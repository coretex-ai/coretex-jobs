import logging

from coretex import CustomDataset, Experiment, Model, MetricType, Metric, folder_manager, currentExperiment

from src.train import train
from src.utils import saveModel
from src.validate import validate
from src.load_data import loadDataAtlas
from src.load_data_std import loadDataStd, prepareForTrainingStd


def validation(experiment: Experiment[CustomDataset]) -> None:
    trainedModelId = experiment.parameters["trainedModel"]

    if trainedModelId is None:
        raise RuntimeError(">> [MicrobiomeForensics] In order to start the validation process You have to type in \"trainedModel\" in experiment parameters")

    logging.info(f">> [MicrobiomeForensics] Fetching pretrained model from Coretex. Model id: {experiment.parameters['trainedModel']}")

    trainedModel = Model.fetchById(trainedModelId)
    trainedModel.download()

    inputMatrix, outputMatrix, sampleIdList, uniqueBodySites, _ = loadDataAtlas(experiment.dataset, experiment)

    validate(experiment, inputMatrix, outputMatrix, uniqueBodySites, sampleIdList, trainedModelId)


def training(experiment: Experiment[CustomDataset]) -> None:
    epochs = experiment.parameters["epochs"]

    experiment.createMetrics([
        Metric.create("loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1])
    ])

    folder_manager.createTempFolder("modelFolder")

    if experiment.parameters["datasetType"] == 1:
        logging.info(">> [MicrobiomeForensics] Standard data selected")

        level, datasetLen, uniqueTaxons, uniqueBodySites = loadDataStd(experiment.dataset, experiment)
        inputMatrix, outputMatrix, sampleIdList = prepareForTrainingStd(level, datasetLen, uniqueTaxons, uniqueBodySites, experiment)

        accuracy = train(experiment, inputMatrix, outputMatrix, uniqueBodySites, uniqueTaxons, sampleIdList)

    else:
        logging.info(">> [MicrobiomeForensics] Raw Microbe Atlas data selected")

        inputMatrix, outputMatrix, sampleIdList, uniqueBodySites, uniqueTaxons = loadDataAtlas(experiment.dataset, experiment)
        datasetLen = inputMatrix.shape[0]

        accuracy = train(experiment, inputMatrix, outputMatrix, uniqueBodySites, uniqueTaxons, sampleIdList)

    saveModel(accuracy, uniqueBodySites, datasetLen, len(uniqueTaxons), experiment)


def main() -> None:
    experiment: Experiment[CustomDataset] = currentExperiment()

    validation(experiment) if experiment.parameters["validation"] else training(experiment)


if __name__ == "__main__":
    main()
