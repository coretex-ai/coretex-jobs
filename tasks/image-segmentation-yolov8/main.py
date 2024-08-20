import logging
import json

from coretex import TaskRun, ImageDataset, currentTaskRun, TaskRunStatus, folder_manager, Model

from src.dataset import isValidationSplitValid, prepareDataset
from src.train import train
from src.validate import validate


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    if taskRun.parameters["validation"]:
        logging.info(">> [Image Segmentation] Validating mode")
        if taskRun.parameters["trainedModel"] is None:
            raise RuntimeError("Model used for image segmentation that needs validation is not valid")

        ctxModel: Model = taskRun.parameters["trainedModel"]
        ctxModel.download()

        with (ctxModel.path / ctxModel.modelDescriptorFileName()).open("r") as file:
            modelDesc = dict(json.load(file))
            imgSize = int(modelDesc["imageSize"])
            classLabels = [str(clazz["label"]) for clazz in modelDesc["labels"]]

        modelPath = ctxModel.path / "best.pt"

        excludedClasses = [label for label in taskRun.dataset.classes.labels if label not in classLabels]
        logging.info(f">> [Image Segmentation] Excluding classes: {excludedClasses}")
        taskRun.dataset.classes.exclude(excludedClasses)

        taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset")
        taskRun.dataset.download()

        validate(taskRun, modelPath, imgSize)
    else:
        logging.info(">> [Image Segmentation] Training mode")
        if taskRun.parameters["epochs"] is None:
            raise RuntimeError(">> [Image Segmentation] The number of training epochs is not defined")
        if taskRun.parameters["imageSize"] is None:
            raise RuntimeError(">> [Image Segmentation] imageSize parameter is not defined")

        excludedClasses = taskRun.parameters["excludedClasses"]
        logging.info(f">> [Image Segmentation] Excluding classes: {excludedClasses}")
        taskRun.dataset.classes.exclude(excludedClasses)

        taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset")
        taskRun.dataset.download()

        if not isValidationSplitValid(taskRun.parameters["validationSplit"], taskRun.dataset.count):
            raise ValueError(f">> [Image Segmentation] validationSplit parameter is invalid")

        datasetPath = folder_manager.createTempFolder("dataset")
        trainDatasetPath, validDatasetPath = prepareDataset(taskRun.dataset, datasetPath, taskRun.parameters["validationSplit"])

        train(taskRun, datasetPath, trainDatasetPath, validDatasetPath)


if __name__ == "__main__":
    main()
