from typing import Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future

import os
import logging
import time
import pickle

from objects import Sample, Taxon

from coretex import folder_manager


def processByteBatch(
    filePath: Path,
    start: int,
    end: int,
    dirPath: Path,
    validTaxons: dict[str, int]
) -> list[Sample]:

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
                    with dirPath.joinpath(currentSampleData.sampleId).open("wb") as file:
                        pickle.dump(currentSampleData, file)

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

                if taxonId not in validTaxons:
                    currentSampleData = None
                    continue

                currentSampleData.addTaxon(Taxon(
                    taxonId,
                    int(count)
                ))

    return sampleData


def loadDataAtlas(
    inputPath: Path,
    modelDir: Path
) -> tuple[Path, dict[str, int], dict[str, int], list[str]]:

    workerCount = os.cpu_count()  # This value should not exceed the total number of CPU cores
    if workerCount is None:
        workerCount = 1

    logging.info(f">> [MicrobiomeForensics] Using {workerCount} CPU cores to read the file")

    fileSize = inputPath.stat().st_size
    # Smaller file size - used for testing
    # fileSize = 100 * 1024 * 1024

    step = fileSize // workerCount
    remainder = fileSize % workerCount

    with open(modelDir / "uniqueTaxons.pkl", "rb") as f:
        uniqueTaxons = pickle.load(f)

    with open(modelDir / "uniqueBodySites.pkl", "rb") as f:
        uniqueBodySites = pickle.load(f)

    def onProcessingFinished(future: Future) -> None:
        exception = future.exception()
        if exception is not None:
            raise exception

    logging.info(f">> [MicrobiomeForensics] Reading: {inputPath}")

    startTime = time.time()
    with ProcessPoolExecutor(max_workers = workerCount) as processPool:
        # Parallel processing is used to read data from file
        datasetPath = folder_manager.createTempFolder("dataset")

        for i in range(workerCount):
            start = i * step
            end = start + step

            future = processPool.submit(
                processByteBatch,
                inputPath,
                start,
                end,
                datasetPath,
                uniqueTaxons
            )
            future.add_done_callback(onProcessingFinished)

        if remainder != 0:
            future = processPool.submit(
                processByteBatch,
                inputPath,
                end,
                end + remainder,
                datasetPath,
                uniqueTaxons
            )
            future.add_done_callback(onProcessingFinished)

    logging.info(f">> [MicrobiomeForensics] Loaded data in: {(time.time() - startTime):.1f}s")

    sampleIdList: list[str] = []
    for path in datasetPath.iterdir():
        sampleIdList.append(path.name)

    return datasetPath, uniqueTaxons, uniqueBodySites, sampleIdList
