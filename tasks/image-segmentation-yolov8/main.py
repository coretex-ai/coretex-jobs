import logging

from coretex import TaskRun, ImageDataset, currentTaskRun, TaskRunStatus

from src.train import train
from src.validate import validate


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    excludedClasses: list[str] = taskRun.parameters["excludedClasses"]
    logging.info(f">> [Image Segmentation] Excluding classes: {excludedClasses}")
    taskRun.dataset.classes.exclude(excludedClasses)

    taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset")
    taskRun.dataset.download()

    if taskRun.parameters["validationMode"]:
        validate(taskRun)
    else:
        train(taskRun)


if __name__ == "__main__":
    main()
