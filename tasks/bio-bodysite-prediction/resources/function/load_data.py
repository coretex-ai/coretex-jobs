from typing import Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future

import os
import logging
import time
import pickle

from scipy import sparse
from sklearn.feature_selection import SelectPercentile

import numpy as np

from objects import Sample, Taxon


def processByteBatch(filePath: Path, start: int, end: int) -> list[Sample]:
    logging.info(f">> [MicrobiomeForensics] Executing process ({start} - {end})")

    sampleData: list[Sample] = []

    with filePath.open("rb") as taxonFile:
        taxonFile.seek(start)

        # The sample currently being read. Once fully read, it will be appended to the list of all samples (SampleData)
        currentSampleData: Optional[Sample] = None

        for line in taxonFile:
            if taxonFile.tell() >= end:
                break

            decodedLine = line.decode("utf-8")

            # ">" indicates the start of a new sample and is followed by the sample ID
            if decodedLine.startswith(">"):
                if currentSampleData is not None:
                    sampleData.append(currentSampleData)
                    currentSampleData = None

                lineData = decodedLine.strip().split("\t")
                sampleId = lineData[0][1:]

                currentSampleData = Sample(
                    sampleId,
                    []
                )
            # If the line does not start with ">" then it holds taxon data
            elif currentSampleData is not None:
                taxonId, count = decodedLine.strip().split("\t")

                currentSampleData.addTaxon(Taxon(
                    taxonId,
                    int(count)
                ))

    return sampleData


def removeBadSamples(sampleData: list[Sample], uniqueTaxons: dict[str, int]) -> list[Sample]:
    sampleDataNew: list[Sample] = []

    for sample in sampleData:
        taxons = sample.taxons

        if all(x.taxonId in uniqueTaxons for x in taxons):
            sampleDataNew.append(sample)

    return sampleDataNew


def selectPercentile(
    inputMatrix: sparse.csr_matrix,
    trainedModelPath: Path,
    percentile: int
) -> sparse.csr_matrix:

    if percentile < 0 or percentile > 100:
        logging.error(">> [MicrobiomeForensics] The percentile parameter was not entered correctly. Continuing as if 100")
        percentile = 100

    percentileFileName = "selectPercentile.pkl"
    percentileFilePath = trainedModelPath / percentileFileName

    if percentileFilePath.exists():
        with open(percentileFilePath , "rb") as file:
            selectPercentile: SelectPercentile = pickle.load(file)

        return selectPercentile.transform(inputMatrix)
    else:
        return inputMatrix


def loadDataAtlas(
    inputPath: Path,
    modelDir: Path,
    percentile: int
) -> tuple[np.ndarray, dict[str, int], list[str]]:

    workerCount = os.cpu_count()  # This value should not exceed the total number of CPU cores
    if workerCount is None:
        workerCount = 1

    logging.info(f">> [MicrobiomeForensics] Using {workerCount} CPU cores to read the file")

    fileSize = inputPath.stat().st_size
    # Smaller file size - used for testing
    fileSize = 100 * 1024 * 1024

    step = fileSize // workerCount
    remainder = fileSize % workerCount

    sampleData: list[Sample] = []

    with open(modelDir / "uniqueTaxons.pkl", "rb") as f:
        uniqueTaxons = pickle.load(f)

    with open(modelDir / "uniqueBodySites.pkl", "rb") as f:
        uniqueBodySites = pickle.load(f)

    def onProcessingFinished(future: Future) -> None:

        """
            The callback function used to assamble the separately read data into complete variables.

            Parameters
            ----------
            future : Future
                The future object of the process from ProcessPoolExecutor
        """

        exception = future.exception()
        if exception is not None:
            raise exception

        processSampleData = future.result()
        sampleData.extend(processSampleData)

    logging.info(f">> [MicrobiomeForensics] Reading: {inputPath}")

    startTime = time.time()
    with ProcessPoolExecutor(max_workers = workerCount) as processPool:
        # Parallel processing is used to read data from file

        for i in range(workerCount):
            start = i * step
            end = start + step

            future = processPool.submit(processByteBatch, inputPath, start, end)
            future.add_done_callback(onProcessingFinished)

        if remainder != 0:
            future = processPool.submit(processByteBatch, inputPath, end, end + remainder)
            future.add_done_callback(onProcessingFinished)

    logging.info(f">> [MicrobiomeForensics] Loaded data in: {(time.time() - startTime):.1f}s")

    sampleData = removeBadSamples(sampleData, uniqueTaxons)

    return prepareForInferenceAtlas(modelDir, sampleData, uniqueTaxons, uniqueBodySites, percentile)


def prepareForInferenceAtlas(
    modelDir: Path,
    mappedSampleObjList: list[Sample],
    uniqueTaxons: dict[str, int],
    uniqueBodySites: dict[str, int],
    percentile: Optional[int]
) -> tuple[np.ndarray, dict[str, int], list[str]]:

    sampleIdList: list[str] = []
    rowIndices: list[int] = []
    columnIndices: list[int] = []
    matrixData: list[int] = []

    for i, sample in enumerate(mappedSampleObjList):

        sampleIdList.append(sample.sampleId)

        # Asseble the input matrix into three lists, with each index between them representing a data point
        for taxon in sample.taxons:
            rowIndices.append(i)   # The sample index (x coordinate / row of the matrix)
            columnIndices.append(uniqueTaxons[taxon.taxonId.rstrip("\x00")])  # The feature (y coordinate / column of the matrix)
            matrixData.append(taxon.count)  # The value that is represented by x and y

    inputMatrixShape = (len(mappedSampleObjList), len(uniqueTaxons))
    inputMatrix = sparse.csr_matrix((matrixData, (rowIndices, columnIndices)), inputMatrixShape)

    if percentile is not None:
        inputMatrix = selectPercentile(
            inputMatrix,
            modelDir,
            percentile
        )

    logging.info(f">> [MicrobiomeForensics] Input matrix shape: {inputMatrix.shape}")


    return inputMatrix, uniqueBodySites, sampleIdList
