import logging
import json

from coretex import TaskRunStatus, ImageDataset, TaskRun, Metric, MetricType, currentTaskRun, Model
from keras import models

from src.train import train
from src.validate import validate




def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    taskRun.createMetrics([
        Metric.create("loss", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]]),
        Metric.create("accuracy", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]], [0, 1]),
        Metric.create("val_loss", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]]),
        Metric.create("val_accuracy", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]], [0, 1])
    ])

    taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset")
    taskRun.dataset.download()




    if taskRun.parameters["validation"]:
        # validating model

        if taskRun.parameters["trainedModel"] is None:
            raise RuntimeError("Model used for image segmentation that needs validation is not valid")

        ctxModel: Model = taskRun.parameters["trainedModel"]
        ctxModel.download()
        with (ctxModel.path / ctxModel.modelDescriptorFileName()).open("r") as file:
            modelDesc = dict(json.load(file))
            #imgSize = modelDesc["imageSize"]
            classLabels = [clazz["label"] for clazz in modelDesc["labels"]]

        excludedClasses = [label for label in taskRun.dataset.classes.labels if label not in classLabels]
        logging.info(f">> [Image Segmentation] Excluding classes: {excludedClasses}")
        taskRun.dataset.classes.exclude(excludedClasses)

        modelPath = ctxModel.path / "tensorflow-model"
        model = models.load_model(modelPath)

        validate(taskRun, model)

    else:
        # training model

        excludedClasses: list[str] = taskRun.parameters["excludedClasses"]
        logging.info(f">> [Image Segmentation] Excluding classes: {excludedClasses}")
        taskRun.dataset.classes.exclude(excludedClasses)

        train(taskRun)



if __name__ == "__main__":
    main()
