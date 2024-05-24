from pathlib import Path

import os
import json
import shutil
import logging

from coretex import currentTaskRun, Model, folder_manager


def copyDir(src: Path, dst: Path, directoryName: str) -> None:
    shutil.copytree(src, dst / directoryName, copy_function = os.link)


def main():
    taskRun = currentTaskRun()
    model = taskRun.parameters["model"]

    ctxModel = Model.createModel(f"{taskRun.id}-{model}-chatbot", taskRun.id, 1.0, {})

    modelFunction = Path(".", "resources", "function")
    resDir = folder_manager.createTempFolder("resDir")
    configPath = resDir / "config.json"

    copyDir(modelFunction, resDir, "function")
    with configPath.open("w") as file:
        json.dump({
            "model": model
        }, file)

    ctxModel.upload(resDir)

    logging.info(">> [Chatbot] Model deployed \U0001F680\U0001F680\U0001F680")


if __name__ == "__main__":
    main()
