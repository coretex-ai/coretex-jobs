from coretex import currentTaskRun, TaskRun, ImageDataset

from src.validation import validation
from src.train import train


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    if taskRun.parameters["validation"]:
        if taskRun.parameters["trainedModel"] is None:
            raise RuntimeError("Model used for image segmentation that needs validation is not valid")
        validation(taskRun, taskRun.parameters["trainedModel"])
    else:
        train(taskRun)


if __name__ == "__main__":
    main()
