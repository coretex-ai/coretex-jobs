import logging

from coretex import CustomDataset, TaskRun, Model, MetricType, Metric, folder_manager, currentTaskRun

from src.train import train
from src.utils import saveModel
from src.validate import validate
from src.load_data import loadDataAtlas
from src.load_data_std import loadDataStd, prepareForTrainingStd


def validation(taskRun: TaskRun[CustomDataset]) -> None:
    trainedModel= taskRun.parameters.get("trainedModel")
    if not isinstance(trainedModel, Model):
        raise RuntimeError(">> [MicrobiomeForensics] In order to start the validation process You have to type in \"trainedModel\" in TaskRun parameters")

    trainedModel.download()

    inputMatrix, outputMatrix, sampleIdList, uniqueBodySites, _ = loadDataAtlas(taskRun.dataset, taskRun)

    validate(taskRun, inputMatrix, outputMatrix, uniqueBodySites, sampleIdList, trainedModel.path)


def training(taskRun: TaskRun[CustomDataset]) -> None:
    epochs = taskRun.parameters["epochs"]

    taskRun.createMetrics([
        Metric.create("loss", "epoch", MetricType.int, "loss", MetricType.float, [0, epochs]),
        Metric.create("accuracy", "epoch", MetricType.int, "accuracy", MetricType.float, [0, epochs], [0, 1])
    ])

    folder_manager.createTempFolder("modelFolder")

    if taskRun.parameters["datasetType"] == 1:
        logging.info(">> [MicrobiomeForensics] Standard data selected")

        level, datasetLen, uniqueTaxons, uniqueBodySites = loadDataStd(taskRun.dataset, taskRun)
        inputMatrix, outputMatrix, sampleIdList = prepareForTrainingStd(level, datasetLen, uniqueTaxons, uniqueBodySites, taskRun)

        accuracy = train(taskRun, inputMatrix, outputMatrix, uniqueBodySites, uniqueTaxons, sampleIdList)

    else:
        logging.info(">> [MicrobiomeForensics] Raw Microbe Atlas data selected")

        inputMatrix, outputMatrix, sampleIdList, uniqueBodySites, uniqueTaxons = loadDataAtlas(taskRun.dataset, taskRun)
        datasetLen = inputMatrix.shape[0]

        accuracy = train(taskRun, inputMatrix, outputMatrix, uniqueBodySites, uniqueTaxons, sampleIdList)

    saveModel(
        accuracy,
        uniqueBodySites,
        datasetLen,
        len(uniqueTaxons),
        taskRun,
        taskRun.parameters["percentile"],
        taskRun.parameters["taxonomicLevel"]
    )


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    validation(taskRun) if taskRun.parameters["validation"] else training(taskRun)


if __name__ == "__main__":
    main()
