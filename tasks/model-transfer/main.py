from typing import Iterator
from contextlib import contextmanager

import logging

from coretex import Model, currentTaskRun, CredentialsSecret
from coretex.networking import networkManager


@contextmanager
def currentUser(username: str, password: str) -> Iterator[None]:
    oldApiToken = networkManager._apiToken
    oldRefreshToken = networkManager._refreshToken

    try:
        response = networkManager.authenticate(username, password, storeCredentials = False)
        if response.hasFailed():
            raise RuntimeError("Failed to authenticate")

        yield None
    finally:
        networkManager._apiToken = oldApiToken
        networkManager._refreshToken = oldRefreshToken


def main() -> None:
    taskRun = currentTaskRun()

    credentials = CredentialsSecret.fetchByName(taskRun.parameters["sourceCredentials"])

    logging.info(">> [Coretex] Downloading source Model")

    # Download source model using provided credentials
    with currentUser(credentials.username, credentials.password):
        sourceModel = Model.fetchById(taskRun.parameters["sourceModelId"])
        sourceModel.download()

    if taskRun.parameters.get("modelName") is not None:
        modelName = taskRun.parameters["modelName"]
    else:
        modelName = sourceModel.name

    logging.info(">> [Coretex] Transfering Model")
    destinationModel = Model.createModel(
        modelName,
        taskRun.id,
        sourceModel.accuracy,
        sourceModel.meta
    )
    destinationModel.upload(sourceModel.path)


if __name__ == "__main__":
    main()
