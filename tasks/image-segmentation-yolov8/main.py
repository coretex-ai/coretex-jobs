import logging
import json

from coretex import TaskRun, ImageDataset, currentTaskRun, TaskRunStatus, folder_manager, Model

from src.dataset import isValidationSplitValid, prepareDataset
from src.train import train
from src.validate import validate


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()
    if not 0 <= taskRun.parameters["confidenceTreshold"] <= 1:
        raise ValueError(f">> [Image Segmentation] The \"confidenceThreshold\" must be between 0 and 1. The current value is: {taskRun.parameters['confidenceTreshold']}.")

    if taskRun.parameters["validation"]:
        logging.info(">> [Image Segmentation] Validating mode")
        if taskRun.parameters.get("trainedModel") is None:
            raise RuntimeError("Model used for image segmentation that needs validation is not valid")

        ctxModel: Model = taskRun.parameters["trainedModel"]
        ctxModel.download()

        with ctxModel.path.joinpath(ctxModel.modelDescriptorFileName()).open("r") as file:
            modelDesc = json.load(file)
            if not isinstance(modelDesc, dict):
                raise ValueError(">> [Image Segmentation] The expected type of objects from the JSON file is \"dict\", but a different type was read.")

            imgSize = modelDesc["imageSize"]
            if not isinstance(imgSize, int):
                raise ValueError(">> [Image Segmentation] The expected type of the object for the \"imageSize\" key in the dictionary from the JSON file is \"int\", but a different type was read.")

            classLabels = [clazz["label"] for clazz in modelDesc["labels"]]
            for label in classLabels:
                if not isinstance(label, str):
                    raise ValueError(">> [Image Segmentation] The expected type of the object for the \"labels\" key in the dictionary from the JSON file is \"dict[str, str]\", but a different type was read.")

        modelPath = ctxModel.path / "best.pt"

        excludedClasses = [label for label in taskRun.dataset.classes.labels if label not in classLabels]
        logging.info(f">> [Image Segmentation] Excluding classes: {excludedClasses}")
        taskRun.dataset.classes.exclude(excludedClasses)

        taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset")
        taskRun.dataset.download()

        validate(taskRun, modelPath, imgSize)
    else:
        logging.info(">> [Image Segmentation] Training mode")
        if taskRun.parameters.get("epochs") is None:
            raise RuntimeError(">> [Image Segmentation] The number of training epochs is not defined")
        if taskRun.parameters.get("imageSize") is None:
            raise RuntimeError(">> [Image Segmentation] The \"imageSize\" parameter is not defined")
        if taskRun.parameters.get("earlyStopping") is None:
            raise RuntimeError(">> [Image Segmentation] The \"earlyStopping\" parameter is not defined")

        excludedClasses = taskRun.parameters["excludedClasses"]
        logging.info(f">> [Image Segmentation] Excluding classes: {excludedClasses}")
        taskRun.dataset.classes.exclude(excludedClasses)

        taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset")
        taskRun.dataset.download()

        if not isValidationSplitValid(taskRun.parameters.get("validationSplit", 0.2), taskRun.dataset.count):
            raise ValueError(f">> [Image Segmentation] validationSplit parameter is invalid")

        datasetPath = folder_manager.createTempFolder("dataset")
        trainDatasetPath, validDatasetPath = prepareDataset(taskRun.dataset, datasetPath, taskRun.parameters["validationSplit"])

        train(taskRun, datasetPath, trainDatasetPath, validDatasetPath)


if __name__ == "__main__":
    main()
