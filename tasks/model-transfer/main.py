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

    logging.info(">> [Coretex] Downloading Model")
    model: Model = taskRun.parameters["model"]
    model.download()

    # Create model in the destination Project with the destination account
    credentials = CredentialsSecret.fetchByName(taskRun.parameters["destinationAccount"])
    credentials = credentials.decrypted()

    with currentUser(credentials.username, credentials.password):
        if taskRun.parameters.get("modelName") is not None:
            modelName = taskRun.parameters["modelName"]
        else:
            modelName = model.name

        logging.info(">> [Coretex] Creating Model...")
        destinationModel = Model.createModel(
            modelName,
            taskRun.parameters["destinationProject"],
            model.accuracy,
            model.meta
        )
        destinationModel.upload(model.path)

    logging.info(">> [Coretex] Model created successfully!")


if __name__ == "__main__":
    main()
