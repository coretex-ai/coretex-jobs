from typing import Optional
from pathlib import Path

import pickle
import zipfile
import logging

from scipy import sparse
from sklearn.feature_selection import SelectPercentile

import numpy as np

from coretex import CustomDataset, CustomSample
from coretex.folder_management import FolderManager
from coretex.utils.hash import hashCachName

from . import cache_filenames as cf
from .objects import MatrixTuple


def getMatrixName(
    datasetName: int,
    sampleOrigin: list[str],
    sequencingTechnique: list[str],
    percentile: int,
    quantize: bool
) -> str:

    origins = ".".join(sampleOrigin)
    techniques = ".".join(sequencingTechnique)

    suffix = f"{origins}_{techniques}_{percentile}_{quantize}"

    return hashCachName(datasetName, suffix)


def loadMatrixCache(cacheName: str, validation: bool) -> MatrixTuple:
    logging.info(">> [MicrobiomeForensics] Loading processed data from cache")

    cache = getMatrixCache(cacheName)
    cache.download()
    cache.samples[0].unzip()
    cachePath = Path(cache.samples[0].path)

    with open(cachePath.joinpath(f"{cf.INPUT_MATRIX}.pkl"), "rb") as f:
        inputMatrix = pickle.load(f)

    with open(cachePath.joinpath(f"{cf.OUTPUT_MATRIX}.pkl"), "rb") as f:
        outputMatrix = pickle.load(f)

    with open(cachePath.joinpath(f"{cf.SAMPLE_ID_LIST}.pkl"), "rb") as f:
        sampleIdList = pickle.load(f)

    with open(cachePath.joinpath(f"{cf.UNIQUE_BODYSITE}.pkl"), "rb") as f:
        uniqueBodySite = pickle.load(f)

    with open(cachePath.joinpath(f"{cf.UNIQUE_TAXONS}.pkl"), "rb") as f:
        uniqueTaxons = pickle.load(f)

    with open(cachePath.joinpath(f"{cf.SELECT_PERCENTILE}.pkl"), "rb") as f:
        selectPercentile = pickle.load(f)

    if not validation:
        modelPath = Path(FolderManager.instance().getTempFolder("modelFolder"))
        with open(modelPath / f"{cf.SELECT_PERCENTILE}.pkl", "wb") as file:
            pickle.dump(selectPercentile, file)

    logging.info(">> [MicrobiomeForensics] Processed data loaded from cache")

    return MatrixTuple(inputMatrix, outputMatrix, sampleIdList, uniqueBodySite, uniqueTaxons)


def cacheMatrix(
    cacheName: str,
    inputMatrix: sparse.csr_matrix,
    outputMatrix: np.ndarray,
    sampleIdList: list[str],
    uniqueBodySite: dict[str, int],
    uniqueTaxons: dict[str, int],
    percentileModel: SelectPercentile,
    spaceId: int
) -> None:

    logging.info(">> [MicrobiomeForensics] Uploading processed data to cache")

    cachePath = Path(FolderManager.instance().temp)
    zipPath = cachePath.joinpath("preparedCache.zip")

    cachedItems = [inputMatrix, outputMatrix, sampleIdList, uniqueBodySite, uniqueTaxons, percentileModel]
    cachedItemsStr = [
        cf.INPUT_MATRIX,
        cf.OUTPUT_MATRIX,
        cf.SAMPLE_ID_LIST,
        cf.UNIQUE_BODYSITE,
        cf.UNIQUE_TAXONS,
        cf.SELECT_PERCENTILE
    ]

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

    if CustomSample().createCustomSample("zipedCache", cacheDataset.id, zipPath):
        logging.info(">> [MicrobiomeForensics] Successfuly cached processed data")
    else:
        logging.warning(">> [MicrobiomeForensics] Failed to cache processed data")


def matrixCacheExists(cacheName: str) -> bool:
    return getMatrixCache(cacheName) is not None


def getMatrixCache(cacheName: str) -> Optional[CustomDataset]:
    """
    Returns the cache CustomDataset object
    """

    cache = CustomDataset.fetchAll(queryParameters = [f"name={cacheName }", "include_sessions=1"])

    return CustomDataset.fetchById(cache[0].id) if len(cache) != 0 else None


def isMatrixCacheValid(cacheName: str) -> bool:
    cache = getMatrixCache(cacheName)
    if cache is None:
        return True

    return len(cache.samples) != 0
