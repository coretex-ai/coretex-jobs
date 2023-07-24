from typing import Any, Optional
from pathlib import Path

import pickle
import zipfile
import logging

from coretex import CustomDataset, CustomSample
from coretex.folder_management import FolderManager
from coretex.utils.hash import hashCachName

from . import cache_filenames as cf
from .objects import JsonTuple, Sample


def getJsonName(datasetName: str, sampleOrigin: list[str], sequencingTechnique: list[str]) -> str:
    origins = ".".join(sampleOrigin)
    techniques = ".".join(sequencingTechnique)

    suffix = f"{origins}_{techniques}"

    return hashCachName(datasetName, suffix)


def loadJsonCache(cacheName: str) -> JsonTuple:
    logging.info(">> [MicrobiomeForensics] Loading assembled dataset from cache")

    cache = getJsonCache(cacheName)
    cache.download()
    cache.samples[0].unzip()
    cachePath = Path(cache.samples[0].path)

    with open(cachePath.joinpath(f"{cf.SAMPLE_DATA}.pkl"), "rb") as f:
        sampleData = pickle.load(f)

    with open(cachePath.joinpath(f"{cf.UNIQUE_TAXONS}.pkl"), "rb") as f:
        uniqueTaxons = pickle.load(f)

    with open(cachePath.joinpath(f"{cf.UNIQUE_BODYSITE}.pkl"), "rb") as f:
        uniqueBodySite = pickle.load(f)

    logging.info(">> [MicrobiomeForensics] Assembled dataset loaded from cache")

    return JsonTuple(sampleData, uniqueTaxons, uniqueBodySite)


def cacheJson(
    cacheName: str,
    sampleData: list[Sample],
    uniqueTaxons: dict[str, int],
    uniqueBodySite: dict[str, int],
    spaceId: int
) -> None:

    logging.info(">> [MicrobiomeForensics] Saving assembled dataset to cache")

    cachePath = Path(FolderManager.instance().temp)
    zipPath = cachePath.joinpath("jsonCache.zip")

    cachedItems = [sampleData, uniqueTaxons, uniqueBodySite]
    cachedItemsStr = [cf.SAMPLE_DATA, cf.UNIQUE_TAXONS, cf.UNIQUE_BODYSITE]
    for i, item in enumerate(cachedItemsStr):
        with open(cachePath.joinpath(f"{item}.pkl"), "wb") as cacheFile:
            pickle.dump(cachedItems[i], cacheFile)

    with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED) as archive:
        for item in cachedItemsStr:
            archive.write(cachePath.joinpath(f"{item}.pkl"), f"{item}.pkl")

    cacheDataset = CustomDataset.createDataset(cacheName, spaceId)
    if cacheDataset is None:
        logging.warning(">> [MicrobiomeForensics] Failed to create cache for processed data")
        return

    if CustomSample.createCustomSample("zipedCache", cacheDataset.id, zipPath):
        logging.info(">> [MicrobiomeForensics] Successfuly cached assembled dataset")
    else:
        logging.info(">> [MicrobiomeForensics] Failed to cache assembled dataset")


def jsonCacheExists(cacheName: str) -> bool:
    return getJsonCache(cacheName) is not None


def getJsonCache(cacheName: str) -> Optional[CustomDataset]:
    """
    Returns the cache CustomDataset object
    """

    cache = CustomDataset.fetchAll(queryParameters = [f"name={cacheName}", "include_sessions=1"])

    return CustomDataset.fetchById(cache[0].id) if len(cache) != 0 else None


def isJsonCacheValid(cacheName: str) -> bool:
    cache = getJsonCache(cacheName)
    if cache is None:
        return True

    return len(cache.samples) != 0
