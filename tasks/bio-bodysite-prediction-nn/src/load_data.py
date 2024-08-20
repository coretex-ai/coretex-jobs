from typing import BinaryIO, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future

import os
import logging
import time
import pickle

from coretex import TaskRun, CustomDataset, TaskRunStatus, folder_manager

from .cache import cacheExists, cacheDataset, loadCache, getCacheName, isCacheValid
from .utils import getBodySite, plots, validateDataset
from .objects import Sample, Taxon


def getDatasetPath(dataset: CustomDataset) -> tuple[Path, Path]:
    sample = dataset.samples[0]
    sample.unzip()

    for filePath in Path(sample.path).iterdir():
        if filePath.suffix == ".mapped":
            mappedPath = filePath
        elif filePath.suffix == ".info":
            infoPath = filePath

    return mappedPath, infoPath


def readEnvInfo(path: Path, sampleOrigin: list[str], sequencingTechniques: list[str]) -> dict[str, str]:
    envInfoData: dict[str, str] = {}

    logging.info(f">> [MicrobiomeForensics] Reading: {path}")
    with path.open("r") as infoFile:
        while line := infoFile.readline():
            info = line.strip().split("\t")
            if len(info) >= 2:
                sampleId, origin, bodySite, technique = info[0], info[1], info[2], info[3]

                if not any(substr.upper() in origin.upper() for substr in sampleOrigin):
                    continue

                if not any(technique.upper() == string.upper() for string in sequencingTechniques):
                    continue

                if len(bodySite.strip()) != 0:
                    envInfoData[sampleId] = str(bodySite) + "," + info[4]

    return envInfoData


def readByteBlockUntilNewLine(file: BinaryIO, blockSize: int) -> Optional[bytes]:
    content = file.read(blockSize)
    if not content:
        return None

    remainder = bytearray()
    while value := file.read(1):
        if value == b"\n" or value == b"\r":
            break

        remainder.extend(value)

    return content + remainder


def processByteBatch(
    envInfoData: dict[str, str],
    filePath: Path,
    start: int,
    end: int,
    dirPath: Path,
    validBodySites: Optional[dict[str, int]] = None,
    validTaxons: Optional[dict[str, int]] = None
) -> tuple[dict[str, int], dict[str, int]]:

    """
        Called as a process by ProcessPoolExecutor for parallel processing.
        Reads raw data from file and creates a file for each sample which will be later
        read by the TF model.

        Parameters
        ----------
        envInfoData : dict[str, str]
            This dictionary holds the bodysites (values) for all samples based on their ids (keys)
        filePath : Path
            The path to the merged file with all sample info
        start : int
            The starting position in the file for this process
        end : int
            This process will only read up to this point
        dirPath : Path
            The path to the directory where the sample files will be created
        validBodySites : Optional[dict[str, int]]
            Dictionary mapping between bodysite names and their encodings (only used during validation
            to remove unwanted samples)
        validTaxons : Optional[dict[str, int]]
            Dictionary mapping between taxon ids and their encodings (only used during validation to
            remove unwanted samples)

        Returns
        -------
        classDistribution : dict[str, int]
            A dictonary mapping each class to the number of times it occurs in the dataset
        taxonDistribution : dict[str, int]
            A dictonary mapping each taxon id to its total count in the dataset
    """

    logging.info(f">> [MicrobiomeForensics] Executing process ({start} - {end})")

    classDistribution: dict[str, int] = {}
    taxonDistribution: dict[str, int] = {}

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

                result = getBodySite(sampleId, envInfoData)
                if result is None:
                    continue

                bodySite, associationSite = result
                bodySite = bodySite.split(";")[0]

                if validBodySites is not None and bodySite not in validBodySites:
                    continue

                if bodySite not in classDistribution:
                    classDistribution[bodySite] = 1
                else:
                    classDistribution[bodySite] += 1

                currentSampleData = Sample(
                    sampleId,
                    bodySite,
                    associationSite,
                    []
                )
            # If the line does not start with ">" then it holds taxon data
            elif currentSampleData is not None:
                taxonId, count = decodedLine.strip().split("\t")

                if validTaxons is not None and taxonId not in validTaxons:
                    classDistribution[currentSampleData.bodySite] -= 1
                    for taxon in currentSampleData.taxons:
                        taxonDistribution[taxon.taxonId] -= taxon.count

                    currentSampleData = None
                    continue

                if taxonId not in taxonDistribution:
                    taxonDistribution[taxonId] = int(count)
                else:
                    taxonDistribution[taxonId] += int(count)

                currentSampleData.addTaxon(Taxon(
                    taxonId,
                    int(count)
                ))

    return classDistribution, taxonDistribution


def loadDataAtlas(
    taskRun: TaskRun[CustomDataset],
    dataset: CustomDataset,
    datasetPath: Path,
    sampleOrigin: list[str],
    sequencingTechnique: list[str],
    useCache: bool,
    validBodySites: Optional[dict[str, int]] = None,
    validTaxons: Optional[dict[str, int]] = None
) -> tuple[dict[str, int], dict[str, int], int]:

    """
        Loads the dataset and returns it ready for training.

        Parameters
        ----------
        dataset: CustomDataset
            The Coretex dataset we are using for the TaskRun
        datasetPath : Path
            A path object tied to the directory where all the samples are stored
            after processing

        Returns
        -------
        uniqueBodySites : dict[str, int]
            Dictionary mapping between bodysite names and their encodings
        uniqueTaxons : dict[str, int]
            Dictionary mapping between taxon ids and their encodings
        datasetLen : int
            The number of samples in the dataset after processing
    """

    cacheName = getCacheName(dataset.name, sampleOrigin, sequencingTechnique)
    if useCache and cacheExists(cacheName) and isCacheValid(cacheName):
        taskRun.updateStatus(TaskRunStatus.inProgress, "Loading assembled dataset from cache")
        return loadCache(taskRun, cacheName)

    logging.info(">> [MicrobiomeForensics] Downloading dataset...")
    taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset...")
    dataset.download()

    validateDataset(dataset)

    taskRun.updateStatus(TaskRunStatus.inProgress, "Loading data from the dataset")
    logging.info(">> [MicrobiomeForensics] Loading data")

    mappedPath, infoPath = getDatasetPath(dataset)
    sampleInfoObj = readEnvInfo(infoPath, sampleOrigin, sequencingTechnique)

    workerCount = os.cpu_count()  # This value should not exceed the total number of CPU cores
    if workerCount is None:
        workerCount = 1

    logging.info(f">> [MicrobiomeForensics] Using {workerCount} CPU cores to read the dataset")

    fileSize = mappedPath.stat().st_size
    # Smaller file size - used for testing
    # fileSize = 100 * 1024 * 1024

    step = fileSize // workerCount
    remainder = fileSize % workerCount

    # These two dictionaries represent the mapping between the names and encoded integers of the bodysites and taxons respectively
    uniqueBodySite: dict[str, int] = {}
    uniqueTaxons: dict[str, int] = {}

    if validBodySites is not None and validTaxons is not None:
        uniqueBodySite = validBodySites
        uniqueTaxons = validTaxons

    classDistribution: dict[str, int] = {}
    taxonDistribution: dict[str, int] = {}

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

        processClassDistribution, processTaxonDistribution = future.result()

        if validBodySites is None and validTaxons is None:
            for bodySite in processClassDistribution:
                if bodySite not in classDistribution:
                    classDistribution[bodySite] = processClassDistribution[bodySite]
                else:
                    classDistribution[bodySite] += processClassDistribution[bodySite]

                if bodySite in uniqueBodySite:
                    continue

                uniqueBodySite[bodySite] = len(uniqueBodySite)

            for taxon in processTaxonDistribution:
                if taxon not in taxonDistribution:
                    taxonDistribution[taxon] = processTaxonDistribution[taxon]
                else:
                    taxonDistribution[taxon] += processTaxonDistribution[taxon]

                if taxon in uniqueTaxons:
                    continue

                uniqueTaxons[taxon] = len(uniqueTaxons)

    logging.info(f">> [MicrobiomeForensics] Reading: {mappedPath}")

    startTime = time.time()
    with ProcessPoolExecutor(max_workers = workerCount) as processPool:
        # Parallel processing is used to read data from file
        for i in range(workerCount):
            start = i * step
            end = start + step

            future = processPool.submit(
                processByteBatch,
                sampleInfoObj,
                mappedPath,
                start,
                end,
                datasetPath,
                validBodySites,
                validTaxons
            )
            future.add_done_callback(onProcessingFinished)

        if remainder != 0:
            future = processPool.submit(
                processByteBatch,
                sampleInfoObj,
                mappedPath,
                end,
                end + remainder,
                datasetPath,
                validBodySites,
                validTaxons
            )
            future.add_done_callback(onProcessingFinished)

    logging.info(f">> [MicrobiomeForensics] Loaded data in: {time.time() - startTime:.1f}s")

    datasetLen = len(list(datasetPath.iterdir()))

    plots(taskRun, classDistribution, taxonDistribution, datasetLen)

    if useCache and isCacheValid(cacheName):
        taskRun.updateStatus(TaskRunStatus.inProgress, "Saving assembled dataset to cache")
        cachePath = folder_manager.createTempFolder("tempCache")
        cacheDataset(
            cacheName,
            datasetPath,
            cachePath,
            classDistribution,
            taxonDistribution,
            taskRun.projectId
        )

    return uniqueBodySite, uniqueTaxons, datasetLen
