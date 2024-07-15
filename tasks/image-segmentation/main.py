from coretex import currentTaskRun, TaskRun, ImageDataset

from src.validation import validation
from src.train import train


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    if taskRun.parameters["validation"]:
        validation(taskRun)
    else:
        train(taskRun)


if __name__ == "__main__":
    main()
