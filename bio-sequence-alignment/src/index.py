from typing import Optional
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

import requests

from coretex import Experiment, CustomDataset, CustomSample
from coretex.folder_management import FolderManager
from coretex.utils.file import isGzip, gzipDecompress
from coretex.utils.hash import hashCacheName

from .utils import indexCommand


def saveCache(cacheName: str, temp: Path, genomeIndexDir: Path, spaceId: int) -> None:
    genomeDataset = CustomDataset.createDataset(
        cacheName,
        spaceId
    )

    if genomeDataset is None:
        raise RuntimeError(">> [Sequence Alignment] Failed to create coretex dataset for indexed genome cache")

    logging.info(">> [Sequence Alignment] Compressing indexed reference genome for upload to coretex")

    zipPath = temp / "genome.zip"
    with ZipFile(zipPath , "w", ZIP_DEFLATED) as archive:
        for filePath in genomeIndexDir.iterdir():
            archive.write(filePath, filePath.name)

    logging.info(">> [Sequence Alignment] Uploading indexed reference genome to coretex for later reuse")

    if CustomSample.createCustomSample(zipPath.name, genomeDataset.id, zipPath) is None:
        raise RuntimeError(">> [Sequence Alignment] Failed to upload indexed genome")

    logging.info(">> [Sequence Alignment] Indexed reference genome has been successfuly uploaded to coretex")


def loadCache(cache: CustomDataset, filename: str) -> Path:
    logging.info(">> [Sequence Alignment] Downloading indexed genome cache")
    cache.download()
    logging.info(">> [Sequence Alignment] Download successful")

    sample = cache.samples[0]
    sample.unzip()

    return Path(sample.path).joinpath(filename)


def isCacheValid(cache: CustomDataset) -> bool:
    if cache is None:
        return True

    return len(cache.samples) != 0


def getCache(cacheName: str) -> Optional[CustomDataset]:
    cacheDatasetList = CustomDataset.fetchAll(queryParameters = [f"name={cacheName}", "include_sessions=1"])
    if len(cacheDatasetList) > 1:
        raise ValueError(">> [Sequence Alignment] Found more then one cache of indexed genome")

    return cacheDatasetList[0] if len(cacheDatasetList) != 0 else None


def downloadGenome(genomeUrl: str, downloadPath: Path, retryCount: int = 0) -> bool:
    logging.info(f">> [Sequence Alignment] Downloading genome from provided url: {genomeUrl}")

    if retryCount >= 3:
        raise RuntimeError(">> [Sequence Alignment] Failed to download reference genome")

    with requests.get(genomeUrl, stream = True) as r:
        r.raise_for_status()

        with open(downloadPath, 'wb') as f:
            for chunk in r.iter_content(chunk_size = 8192):
                f.write(chunk)

        if isGzip(downloadPath):
            genomePath = Path(str(downloadPath).replace(".gz", ""))
            gzipDecompress(downloadPath, genomePath)

        logging.info(">> [Sequence Alignment] Genome has been successfuly downloaded")

        return r.ok


def index(experiment: Experiment[CustomDataset]) -> Path:
    genomeUrl: str = experiment.parameters["genomeUrl"]

    temp = Path(FolderManager.instance().temp)
    genomeIndexDir = Path(FolderManager.instance().createTempFolder("genome"))

    downloadPath = temp / genomeUrl.split("/")[-1]
    filename = downloadPath.stem if downloadPath.suffix == ".gz" else downloadPath.name

    cacheName = hashCacheName(filename, genomeUrl)
    cache = getCache(cacheName)
    if cache is not None and isCacheValid(cache):
        return loadCache(cache, filename)
    # After dataset deletion on coretex is activated, a check to see if the cache has
    # been unsuccessfuly created on a previous run (dataset has 0 samples) should be conducted
    # and that dataset removed if true to allow for the creation of a functional cache.
    # For now we use isCacheValid to ignore unsuccessfuly created cache

    genomePath = temp / filename
    retryCount = 0

    while not downloadGenome(genomeUrl, downloadPath, retryCount):
        retryCount += 1

    logging.info(">> [Sequence Alignment] Starting reference genome indexing with BWA. This may take a while")

    genomePrefix = genomeIndexDir / filename
    indexCommand(genomePath, genomePrefix)

    logging.info(">> [Sequence Alignment] Reference genome succesfully indexed")

    if isCacheValid(cache):
        saveCache(cacheName, temp, genomeIndexDir, experiment.spaceId)

    return genomePrefix
