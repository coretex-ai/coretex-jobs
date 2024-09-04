from pathlib import Path

import os
import json
import shutil
import logging

from coretex import currentTaskRun, Model, folder_manager, CustomDataset


def copyDir(src: Path, dst: Path, directoryName: str) -> None:
    shutil.copytree(src, dst / directoryName, copy_function = os.link)


def getIndexPath(dataset: CustomDataset) ->  Path:
    sample = dataset.samples[0]
    sample.unzip()

    return sample.path / "embeddings.index"


def main() -> None:
    taskRun = currentTaskRun()

    model = Model.createModel(f"{taskRun.id}-rag-chatbot", taskRun.projectId, 1.0)

    modelFunction = Path(".", "resources", "function")
    resourcesDir = folder_manager.createTempFolder("resourcesDir")

    copyDir(modelFunction, resourcesDir, "function")

    if taskRun.parameters["dataset"] is not None:
        taskRun.dataset.download()
        sample = taskRun.dataset.samples[0]
        sample.unzip()

        copyDir(sample.path, resourcesDir, "corpus-index")

    with resourcesDir.joinpath("metadata.json").open("w") as file:
        json.dump({"streaming": taskRun.parameters["streaming"]}, file)

    model.upload(resourcesDir)

    logging.info(">> [DocumentOCR] Model deployed \U0001F680\U0001F680\U0001F680")


if __name__ == "__main__":
    main()
