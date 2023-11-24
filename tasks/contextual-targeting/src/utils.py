from zipfile import ZipFile
from pathlib import Path

from coretex import cache, folder_manager


def fetchModelFile(modelUrl: str, modelName: str, modelSuffix: str) -> Path:
    if not cache.exists(modelUrl):
        cache.storeUrl(modelUrl, modelName)

    modelPath = cache.getPath(modelUrl)

    with ZipFile(modelPath, "r") as zipFile:
        zipFile.extractall(folder_manager.cache)

    return modelPath.with_suffix(modelSuffix)
