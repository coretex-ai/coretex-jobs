from typing import Optional
from pathlib import Path

import logging
import zipfile
import pickle
import time

from coretex import Run, CustomDataset, CustomSample, folder_manager
from coretex.utils.hash import hashCacheName

from .utils import plots


def getCacheName(datasetName: str, sampleOrigin: list[str], sequencingTechnique: list[str]) -> str:
    origins = ".".join(sampleOrigin)
    techniques = ".".join(sequencingTechnique)

    suffix = f"{origins}_{techniques}"

    return hashCacheName(datasetName + "_nn", suffix)


def getCache(cacheName: str) -> Optional[CustomDataset]:
    cache = CustomDataset.fetchAll(queryParameters = [f"name={cacheName}", "include_sessions=1"])
    return CustomDataset.fetchById(cache[0].id) if len(cache) != 0 else None


def cacheExists(cacheName: str) -> bool:
    return getCache(cacheName) is not None


def cacheDataset(
    cacheName: str,
    datasetPath: Path,
    cachePath: Path,
    classDistribution: dict[str, int],
    taxonDistribution: dict[str, int],
    spaceId: int
) -> None:

    logging.info(">> [MicrobiomeForensics] Saving assembled dataset to cache (this may take a while)")

    cacheDataset = CustomDataset.createDataset(cacheName, spaceId)
    if cacheDataset is None:
        raise RuntimeError(">> [MicrobiomeForensics] Failed to create coretex dataset for cache")

    cachedItems = [taxonDistribution, classDistribution]
    cachedItemNames = ["taxonDistribution", "classDistribution"]
    for cachedItem, cachedItemName in zip(cachedItems, cachedItemNames):
        picklePath = folder_manager.temp / cachedItemName
        with picklePath.open("wb") as cacheFile:
            pickle.dump(cachedItem, cacheFile)

        zipPath = cachePath / f"{cachedItemName}.zip"
        with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(picklePath, picklePath.name)

    for path in datasetPath.iterdir():
        zipPath = cachePath / f"{path.name}.zip"
        with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(path, path.name)

    for path in cachePath.iterdir():
        customSample = CustomSample.createCustomSample(path.name[:-4], cacheDataset.id, path)
        if customSample is None:
            raise RuntimeError(">> [MicrobiomeForensics] Failed to upload cache")

    logging.info(">> [MicrobiomeForensics] Successfuly cached assembled dataset")


def loadCache(run: Run[CustomDataset], cacheName: str) -> tuple[Path, dict[str, int], dict[str, int]]:
    logging.info(">> [MicrobiomeForensics] Loading assembled dataset to cache")
    start = time.time()

    datasetPath = folder_manager.temp / "processedDataset"
    datasetPath.mkdir(parents = True, exist_ok = True)

    cache = getCache(cacheName)
    cache.download()

    samples = cache.getSamples(lambda sample: sample.name != "taxonDistribution" and sample.name != "classDistribution")
    for sample in samples:
        sample.unzip()

        with sample.joinPath(sample.name).open("rb") as file:
            content = pickle.load(file)

        with datasetPath.joinpath(sample.name).open("wb") as file:
            pickle.dump(content, file)

    taxonDistribution = cache.getSample("taxonDistribution")
    classDistribution = cache.getSample("classDistribution")

    if taxonDistribution is None and classDistribution is None:
        raise RuntimeError(">> [MicrobiomeForensics] Could not find taxonDistribution and classDistribution files in cache")
    elif taxonDistribution is None:
        raise RuntimeError(">> [MicrobiomeForensics] Could not find taxonDistribution file in cache")
    elif classDistribution is None:
        raise RuntimeError(">> [MicrobiomeForensics] Could not find classDistribution file in cache")

    uniqueTaxons = generateTaxonEncoding(taxonDistribution)
    uniqueBodySites = generateClassEncodings(classDistribution)

    logging.info(f">> [MicrobiomeForensics] Loaded data in {time.time() - start:.1f}s")

    datasetLen = len(list(datasetPath.iterdir()))

    plots(run, classDistribution, taxonDistribution, datasetLen)

    return datasetPath, uniqueBodySites, uniqueTaxons, datasetLen


def generateTaxonEncoding(taxonDistribution: dict[str, int]) -> dict[str, int]:
    uniqueTaxons: dict[str, int] = {}
    for taxon in taxonDistribution:
        uniqueTaxons[taxon] = len(uniqueTaxons)

    return uniqueTaxons


def generateClassEncodings(classDistribution: dict[str, int]) -> dict[str, int]:
    uniqueBodySites: dict[str, int] = {}
    for bodySite in classDistribution:
        uniqueBodySites[bodySite] = len(uniqueBodySites)

    return uniqueBodySites


def isCacheValid(cacheName: str) -> bool:
    cache = getCache(cacheName)
    if cache is None:
        return True

    return len(cache.samples) != 0
