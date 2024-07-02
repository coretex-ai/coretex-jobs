from typing import Any, Optional
from pathlib import Path

import pickle
import zipfile
import logging

from coretex import CustomDataset, CustomSample, folder_manager, createDataset
from coretex.utils.hash import hashCacheName

from . import cache_filenames as cf
from .objects import JsonTuple, Sample


def getJsonName(datasetName: str, sampleOrigin: list[str], sequencingTechnique: list[str]) -> str:
    origins = ".".join(sampleOrigin)
    techniques = ".".join(sequencingTechnique)

    suffix = f"{origins}-{techniques}"

    return hashCacheName(datasetName, suffix)[:20]


def loadJsonCache(cacheName: str) -> JsonTuple:
    logging.info(">> [MicrobiomeForensics] Loading assembled dataset from cache")

    cache = getJsonCache(cacheName)
    if cache is None:
        raise ValueError(">> [MicrobiomeForensics] Failed to retrieve cache")

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
    projectId: int
) -> None:

    logging.info(">> [MicrobiomeForensics] Saving assembled dataset to cache")

    cachePath = folder_manager.temp
    zipPath = cachePath.joinpath("jsonCache.zip")

    cachedItems = [sampleData, uniqueTaxons, uniqueBodySite]
    cachedItemsStr = [cf.SAMPLE_DATA, cf.UNIQUE_TAXONS, cf.UNIQUE_BODYSITE]
    for i, item in enumerate(cachedItemsStr):
        with open(cachePath.joinpath(f"{item}.pkl"), "wb") as cacheFile:
            pickle.dump(cachedItems[i], cacheFile)

    with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED) as archive:
        for item in cachedItemsStr:
            archive.write(cachePath.joinpath(f"{item}.pkl"), f"{item}.pkl")

    with createDataset(CustomDataset, cacheName, projectId) as cacheDataset:
        try:
            cacheDataset.add(zipPath)
            logging.info(">> [MicrobiomeForensics] Successfuly cached assembled dataset")
        except:
            logging.info(">> [MicrobiomeForensics] Failed to cache assembled dataset")


def jsonCacheExists(cacheName: str) -> bool:
    return getJsonCache(cacheName) is not None


def getJsonCache(cacheName: str) -> Optional[CustomDataset]:
    """
        Returns the cache CustomDataset object
    """

    cache = CustomDataset.fetchAll(
        name = cacheName,
        include_sessions = 1
    )

    return cache[0] if len(cache) != 0 else None


def isJsonCacheValid(cacheName: str) -> bool:
    cache = getJsonCache(cacheName)
    if cache is None:
        return True

    return len(cache.samples) != 0
