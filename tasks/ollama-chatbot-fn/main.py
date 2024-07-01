from pathlib import Path

import os
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

    model = Model.createModel(f"{taskRun.id}-rag-chatbot", taskRun.id, 1.0, {})

    modelFunction = Path(".", "resources", "function")
    resDir = folder_manager.createTempFolder("resDir")

    copyDir(modelFunction, resDir, "function")

    if taskRun.parameters["dataset"] is not None:
        taskRun.dataset.download()
        sample = taskRun.dataset.samples[0]
        sample.unzip()

        copyDir(sample.path, resDir, "corpus-index")

    model.upload(resDir)

    logging.info(">> [DocumentOCR] Model deployed \U0001F680\U0001F680\U0001F680")


if __name__ == "__main__":
    main()
