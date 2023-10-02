import logging
import pickle

from coretex import CustomDataset, TaskRun, Model, MetricType, Metric, folder_manager, currentTaskRun

from src.train import train
from src.utils import saveModel
from src.validate import validate
from src.load_data import loadDataAtlas
from src.load_data_std import loadDataStd


def validation(taskRun: TaskRun[CustomDataset]) -> None:
    logging.info(">> [MicrobiomeForensics] Fetching dataset for validation")

    trainedModel: Model = taskRun.parameters["trainedModel"]
    if trainedModel is None:
        raise RuntimeError(">> [MicrobiomeForensics] In order to start the validation process You have to type in \"trainedModel\" in TaskRun parameters")

    datasetPath = folder_manager.createTempFolder("processedDataset")
    trainedModel.download()

    # The same dictionaries will be used as during training
    with trainedModel.path.joinpath("uniqueBodySites.pkl").open("rb") as f:
        validBodysites = pickle.load(f)

    with trainedModel.path.joinpath("uniqueTaxons.pkl").open("rb") as f:
        validTaxons = pickle.load(f)

    if taskRun.parameters["datasetType"] == 1:
        taxonomicLevel = taskRun.parameters["taxonomicLevel"]
        uniqueBodySites, uniqueTaxons, _ = loadDataStd(
            taskRun,
            taskRun.dataset,
            datasetPath,
            taxonomicLevel,
            validBodysites,
            validTaxons
        )
    else:
        sampleOrigin = taskRun.parameters["sampleOrigin"]
        sequencingTechnique = taskRun.parameters["sequencingTechnique"]
        useCache = taskRun.parameters["cache"]

        uniqueBodySites, uniqueTaxons, _ = loadDataAtlas(
            taskRun,
            taskRun.dataset,
            datasetPath,
            sampleOrigin,
            sequencingTechnique,
            useCache,
            validBodysites,
            validTaxons
        )

    validate(taskRun, datasetPath, uniqueBodySites, uniqueTaxons, trainedModel.path)


def training(taskRun: TaskRun[CustomDataset]) -> None:
    epochs = taskRun.parameters["epochs"]

    taskRun.createMetrics([
        Metric.create("train_loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("valid_loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("train_accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1]),
        Metric.create("valid_accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1])
    ])

    folder_manager.createTempFolder("modelFolder")
    datasetPath = folder_manager.createTempFolder("processedDataset")

    if taskRun.parameters["datasetType"] == 1:
        logging.info(">> [MicrobiomeForensics] Standard data selected")

        taxonomicLevel = taskRun.parameters["taxonomicLevel"]

        uniqueBodySites, uniqueTaxons, datasetLen = loadDataStd(taskRun, taskRun.dataset, datasetPath, taxonomicLevel)
        accuracy = train(taskRun, datasetPath, uniqueBodySites, uniqueTaxons)

    else:
        logging.info(">> [MicrobiomeForensics] Raw Microbe Atlas data selected")

        sampleOrigin = taskRun.parameters["sampleOrigin"]
        sequencingTechnique = taskRun.parameters["sequencingTechnique"]
        useCache = taskRun.parameters["cache"]

        uniqueBodySites, uniqueTaxons, datasetLen = loadDataAtlas(
            taskRun,
            taskRun.dataset,
            datasetPath,
            sampleOrigin,
            sequencingTechnique,
            useCache
        )

        accuracy = train(taskRun, datasetPath, uniqueBodySites, uniqueTaxons)

    saveModel(taskRun, accuracy, uniqueBodySites, datasetLen, len(uniqueTaxons))


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    validation(taskRun) if taskRun.parameters["validation"] else training(taskRun)


if __name__ == "__main__":
    main()
