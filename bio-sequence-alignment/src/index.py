from typing import Optional
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

import requests

from coretex import TaskRun, CustomDataset, CustomSample, folder_manager
from coretex.utils.file import isGzip, gzipDecompress
from coretex.utils.hash import hashCacheName
from coretex.bioinformatics import sequence_alignment as sa

from .filepaths import BWA


GENOME_SUFFIXES = [".fasta", ".fna", ".fa"]


def saveCache(cacheName: str, temp: Path, genomeIndexDir: Path, projectId: int) -> None:
    genomeDataset = CustomDataset.createDataset(
        cacheName,
        projectId
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


def loadCache(cache: CustomDataset) -> Path:
    logging.info(">> [Sequence Alignment] Downloading indexed genome cache")
    cache.download()
    logging.info(">> [Sequence Alignment] Download successful")

    sample = cache.samples[0]
    sample.unzip()

    genomeIndexFilePath = list(Path(sample.path).iterdir())[0]

    return genomeIndexFilePath.parent / genomeIndexFilePath.stem


def isCacheValid(cache: CustomDataset) -> bool:
    return len(cache.samples) != 0


def getCache(cacheName: str) -> Optional[CustomDataset]:
    cacheDatasetList = CustomDataset.fetchAll(
        name = cacheName,
        include_sessions = 1
    )

    if len(cacheDatasetList) > 1:
        logging.warning(">> [Sequence Alignment] Found more then one cache of indexed genome")

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


def loadGenome(dataset: CustomDataset) -> Path:
    dataset.download()
    sample = dataset.samples[0]
    sample.unzip()

    for path in sample.path.iterdir():
        if any([path.suffix == genomeSuffix for genomeSuffix in GENOME_SUFFIXES]):
            return path

    raise ValueError


def index(taskRun: TaskRun[CustomDataset]) -> Path:
    genomeIndexDir = folder_manager.createTempFolder("genome")

    genomeUrl: Optional[str] = taskRun.parameters["genomeUrl"]
    if genomeUrl is not None:
        downloadPath = folder_manager.temp / genomeUrl.split("/")[-1]
        filename = downloadPath.stem if downloadPath.suffix == ".gz" else downloadPath.name
        cacheName = hashCacheName(filename, genomeUrl)
    else:
        referenceDataset = taskRun.parameters["referenceDataset"]
        cacheName = f"{referenceDataset.id}_genomeCache"

    cache = getCache(cacheName)
    if cache is not None and isCacheValid(cache):
        return loadCache(cache)

    if genomeUrl is not None:
        retryCount = 0
        while not downloadGenome(genomeUrl, downloadPath, retryCount):
            retryCount += 1

        genomePath = folder_manager.temp / filename
    else:
        genomePath = loadGenome(referenceDataset)
        filename = genomePath.name

    logging.info(">> [Sequence Alignment] Starting reference genome indexing with BWA. This may take a while")

    genomePrefix = genomeIndexDir / filename
    sa.indexCommand(Path(BWA), genomePath, genomePrefix)

    logging.info(">> [Sequence Alignment] Reference genome succesfully indexed")

    if cache is None or not isCacheValid(cache):
        saveCache(cacheName, folder_manager.temp, genomeIndexDir, taskRun.projectId)

    return genomePrefix
