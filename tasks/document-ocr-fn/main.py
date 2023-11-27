from pathlib import Path

import os
import shutil

from coretex import currentTaskRun, Model, folder_manager


def copyDir(src: Path, dst: Path, directoryName: str) -> None:
    shutil.copytree(src, dst / directoryName, copy_function = os.link)


def main() -> None:
    taskRun = currentTaskRun()
    modelsDir = folder_manager.createTempFolder("models")

    segmentationModel: Model = taskRun.parameters["segmentationModel"]
    objectDetectionModel: Model = taskRun.parameters["objectDetectionModel"]

    segmentationModel.download()
    objectDetectionModel.download()

    modelFunction = Path(".", "resources", "function")
    copyDir(modelFunction, modelsDir, "function")
    copyDir(segmentationModel.path, modelsDir, "segmentationModel")
    copyDir(objectDetectionModel.path, modelsDir, "objectDetectionModel")

    model = Model.createModel(taskRun.name, taskRun.id, objectDetectionModel.accuracy, {})
    model.upload(modelsDir)


if __name__ == "__main__":
    main()
