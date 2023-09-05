from typing import BinaryIO, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future

import os
import logging
import time
import pickle

from scipy import sparse
from sklearn.feature_selection import SelectPercentile, f_classif

import numpy as np

from coretex import Experiment, CustomDataset, ExperimentStatus, folder_manager

from .cache_json import loadJsonCache, jsonCacheExists, cacheJson, isJsonCacheValid, getJsonName
from .cache_matrix import loadMatrixCache, matrixCacheExists, cacheMatrix, isMatrixCacheValid, getMatrixName
from .utils import getBodySite, plots, validateDataset
from .objects import Sample, Taxon, MatrixTuple, JsonTuple


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
    with open(path, "r") as infoFile:
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


def processByteBatch(envInfoData: dict[str, str], filePath: Path, start: int, end: int) -> JsonTuple:

    """
        Called as a process by ProcessPoolExecutor for parallel processing.
        Reads raw data from file and transforms it into a list of Sample objects.

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

        Returns
        -------
        sampleData, uniqueBodySites, uniqueTaxons : JsonTuple
            sampleData : list[Sample]
                List of samples read by this process
            uniqueTaxons : dict[str, int]
                Dictionary mapping between taxon ids and their encodings for this process
            uniqueBodySites : dict[str, int]
                Dictionary mapping between bodysite names and their encodings for this process
    """

    logging.info(f">> [MicrobiomeForensics] Executing process ({start} - {end})")

    sampleData: list[Sample] = []
    uniqueBodySites: set[str] = set()
    uniqueTaxons: set[str] = set()

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

                result = getBodySite(sampleId, envInfoData)
                if result is None:
                    continue

                bodySite, associationSite = result
                bodySite = bodySite.split(";")[0]
                uniqueBodySites.add(bodySite)

                currentSampleData = Sample(
                    sampleId,
                    bodySite,
                    associationSite,
                    []
                )
            # If the line does not start with ">" then it holds taxon data
            elif currentSampleData is not None:
                taxonId, count = decodedLine.strip().split("\t")
                uniqueTaxons.add(taxonId)

                currentSampleData.addTaxon(Taxon(
                    taxonId,
                    int(count)
                ))

    return JsonTuple(sampleData, uniqueBodySites, uniqueTaxons)


def removeBadSamples(sampleData: list[Sample], uniqueTaxons: dict[str, int], uniqueBodySites: dict[str, int]) -> list[Sample]:

    """
        Before validation, this function checks for samples with taxons and bodysites not encountered during training and removes them.

        Parameters
        ----------
        sampleData: list[Sample]
            The list of all samples (the dataset)
        uniqueTaxons : dict[str, int]
            Dictionary mapping between taxon ids and their encodings
        uniqueBodySites : dict[str, int]
            Dictionary mapping between bodysite names and their encodings

        Returns
        -------
        sampleDataNew : list[Sample]
            The sampleData list with unwanted samples removed
    """

    sampleDataNew: list[Sample] = []

    for sample in sampleData:
        taxons = sample.taxons

        if(all(x.taxonId in uniqueTaxons for x in taxons)) and sample.bodySite in uniqueBodySites:
            sampleDataNew.append(sample)

    return sampleDataNew


def removeRareBodySites(sampleData: list[Sample], uniqueBodySites: dict[str, int]) -> tuple[list[Sample], dict[str, int]]:

    """
        Removes bodysites with only a single occurence and the samples associated with them.

        Parameters
        ----------
        sampleData: list[Sample]
            The list of all samples (the dataset)
        uniqueBodySites : dict[str, int]
            Dictionary mapping between bodysite names and their encodings

        Returns
        -------
        sampleData : list[Sample]
            The sampleData list with bad samples removed
        newUniqueBodysites : dict[str, int]
            The dictionary with single occurence bodysites removed
    """

    sitesForRemoval: list = []
    for bodySite in uniqueBodySites:
        count = 0

        for i, sample in enumerate(sampleData):
            if bodySite == sample.bodySite:
                count += 1
                index = i

        if count <= 1:
            sampleData.pop(index)
            sitesForRemoval.append(bodySite)

    for bodySite in sitesForRemoval:
        uniqueBodySites.pop(bodySite)

    newUniqueBodySites: dict[str, int] = {}
    for bodySite in uniqueBodySites:
        newUniqueBodySites[bodySite] = len(newUniqueBodySites)

    return sampleData, newUniqueBodySites


def selectPercentile(
    inputMatrix: sparse.csr_matrix,
    outputMatrix: np.ndarray,
    percentile: int,
    validate: bool,
    trainedModelId: Optional[int]
) -> tuple[sparse.csr_matrix, Optional[SelectPercentile]]:

    """
        Using sklearns SelectPercentile, the total number of features in sampleData are
        reduced to the passed percentile based on a scoring function.

        Parameters
        ----------
        inputMatrix : csr_matrix
            The X (input) of the dataset in sparse matrix form
        outputMatrix : ndarray
            The y vector (labels) of the dataset as a numpy array
        percentile : int
            An integer form 1 to 100 representing the percentage of features that will remain in the dataset
        validate : bool
            Whether to create a new SelectPercentile model (if False) in case of training
            or use the model during training (if True)
        trainedModelId : Optional[int]
            Only needed if validate is True. The id of the model we are using for validation

        Returns
        -------
        inputMatrix : csr_matrix
            The X matrix with features removed
        selectPercentile : Optional[SelectPercentile]
            The feature selection model created if validate is False
    """

    if percentile < 0 or percentile > 100:
        logging.error(">> [MicrobiomeForensics] The percentile parameter was not entered correctly. Continuing as if 100")
        percentile = 100

    percentileFileName = "selectPercentile.pkl"

    if validate:
        trainedModelPath = folder_manager.modelsFolder / str(trainedModelId) / percentileFileName

        if trainedModelPath.exists():
            with open(trainedModelPath , "rb") as file:
                selectPercentile: SelectPercentile = pickle.load(file)

            return selectPercentile.transform(inputMatrix), None
        else:
            return inputMatrix, None

    modelPath = folder_manager.temp / "modelFolder"

    selectPercentile = SelectPercentile(score_func = f_classif, percentile = percentile)
    selectPercentile = selectPercentile.fit(inputMatrix, outputMatrix)
    inputMatrix = selectPercentile.transform(inputMatrix)

    with open(modelPath / percentileFileName, "wb") as file:
        pickle.dump(selectPercentile, file)

    return inputMatrix, selectPercentile


def loadDataAtlas(
    dataset: CustomDataset,
    experiment: Experiment[CustomDataset]
) -> MatrixTuple:

    """
        Loads the dataset and returns it ready for training.

        Parameters
        ----------
        dataset: CustomDataset
            The Coretex dataset we are using for the experiment

        Returns
        -------
        inputMatrix, outputMatrix, sampleIdList, uniqueBodySite, uniqueTaxons : MatrixTuple
            inputMatrix : csr_matrix
                The X (input) of the dataset in sparse matrix form
            outputMatrix : ndarray
                The y vector (labels) of the dataset as a numpy array
            uniqueBodySites : dict[str, int]
                Dictionary mapping between bodysite names and their encodings
            uniqueTaxons : dict[str, int]
                Dictionary mapping between taxon ids and their encodings
    """

    sampleOrigin = experiment.parameters["sampleOrigin"]
    sequencingTechnique = experiment.parameters["sequencingTechnique"]
    useCache = experiment.parameters["cache"]
    validate = experiment.parameters["validation"]

    cacheNameMatrix = getMatrixName(
        dataset.name,
        sampleOrigin,
        sequencingTechnique,
        experiment.parameters["percentile"],
        experiment.parameters["quantize"]
    )

    # Level 2 cache of the fully processed data ready for training
    if useCache and matrixCacheExists(cacheNameMatrix) and isMatrixCacheValid(cacheNameMatrix):
        experiment.updateStatus(ExperimentStatus.inProgress, "Loading processed data from cache")
        return loadMatrixCache(cacheNameMatrix, validate)

    cacheNameJson = getJsonName(
        dataset.name,
        sampleOrigin,
        sequencingTechnique
    )

    # Level 1 cache of the assambled dataset into Sample objects
    if useCache and jsonCacheExists(cacheNameJson) and isJsonCacheValid(cacheNameJson):
        experiment.updateStatus(ExperimentStatus.inProgress, "Loading assembled dataset from cache")
        sampleData, uniqueTaxons, uniqueBodySite = loadJsonCache(cacheNameJson)
        return prepareForTrainingAtlas(experiment, sampleData, uniqueTaxons, uniqueBodySite, cacheNameMatrix)

    logging.info(">> [MicrobiomeForensics] Downloading dataset...")
    experiment.updateStatus(ExperimentStatus.inProgress, "Downloading dataset...")
    dataset.download()

    validateDataset(dataset)

    experiment.updateStatus(ExperimentStatus.inProgress, "Loading data from the dataset")
    logging.info(">> [MicrobiomeForensics] Loading data")

    mappedPath, infoPath = getDatasetPath(dataset)
    sampleInfoObj = readEnvInfo(infoPath, sampleOrigin, sequencingTechnique)

    workerCount = os.cpu_count()  # This value should not exceed the total number of CPU cores
    logging.info(f">> [MicrobiomeForensics] Using {workerCount} CPU cores to read the dataset")

    fileSize = mappedPath.stat().st_size
    # Smaller file size - used for testing
    # fileSize = 100 * 1024 * 1024

    step = fileSize // workerCount
    remainder = fileSize % workerCount

    sampleData: list[Sample] = []

    # These two dictionaries represent the mapping between the names and encoded integers of the bodysites and taxons respectively
    uniqueBodySite: dict[str, int] = {}
    uniqueTaxons: dict[str, int] = {}

    if validate:
        # In the case of validation the same dictionaries will be used as during training
        modelPath = folder_manager.modelsFolder / str(experiment.parameters["trainedModel"])

        with open(modelPath / "uniqueTaxons.pkl", "rb") as f:
            uniqueTaxons = pickle.load(f)

        with open(modelPath / "uniqueBodySites.pkl", "rb") as f:
            uniqueBodySite = pickle.load(f)

    def onProcessingFinished(future: Future) -> None:

        """
            The callback function used to assamble the separately read data into complete variables.

            Parameters
            ----------
            future : Future
                The future object of the process from ProcessPoolExecutor
        """

        if future.exception() is not None:
            raise future.exception()

        processSampleData, processUniqueBodySite, processUniqueTaxons = future.result()

        sampleData.extend(processSampleData)

        if not validate:
            for bodySite in processUniqueBodySite:
                if bodySite in uniqueBodySite:
                    continue

                uniqueBodySite[bodySite] = len(uniqueBodySite)

            for taxon in processUniqueTaxons:
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

            future = processPool.submit(processByteBatch, sampleInfoObj, mappedPath, start, end)
            future.add_done_callback(onProcessingFinished)

        if remainder != 0:
            future = processPool.submit(processByteBatch, sampleInfoObj, mappedPath, end, end + remainder)
            future.add_done_callback(onProcessingFinished)

    if not validate:
        sampleData, uniqueBodySite = removeRareBodySites(sampleData, uniqueBodySite)

    logging.info(f">> [MicrobiomeForensics] Loaded data in: {(time.time() - startTime):.1f}s")

    if useCache and isJsonCacheValid(cacheNameJson):
        experiment.updateStatus(ExperimentStatus.inProgress, "Saving assembled dataset to cache")
        cacheJson(cacheNameJson, sampleData, uniqueTaxons, uniqueBodySite, experiment.spaceId)

    if validate:
        sampleData = removeBadSamples(sampleData, uniqueTaxons, uniqueBodySite)

    return prepareForTrainingAtlas(experiment, sampleData, uniqueTaxons, uniqueBodySite, cacheNameMatrix)


def prepareForTrainingAtlas(
    experiment: Experiment[CustomDataset],
    mappedSampleObjList: list[Sample],
    uniqueTaxons: dict[str, int],
    uniqueBodySite: dict[str, int],
    cacheNameMatrix: str
) -> MatrixTuple:

    """
        Transforms the list of Sample objects to the input and label matrices used for training.

        Parameters
        ----------
        mappedSampleObjList : list[Sample]
            The list of all samples in Sample object format
        uniqueTaxons : dict[str, int]
            Dictionary mapping between taxon ids and their encodings
        uniqueBodySite : dict[str, int]
            Dictionary mapping between bodysite names and their encodings

        Returns
        -------
        inputMatrix, outputMatrix, sampleIdList, uniqueBodySite, uniqueTaxons : MatrixTuple
            inputMatrix : csr_matrix
                The X (input) of the dataset in sparse matrix form
            outputMatrix : ndarray
                The y vector (labels) of the dataset as a numpy array
            uniqueBodySites : dict[str, int]
                Dictionary mapping between bodysite names and their encodings
            uniqueTaxons : dict[str, int]
                Dictionary mapping between taxon ids and their encodings
    """

    quantize = experiment.parameters["quantize"]

    experiment.updateStatus(ExperimentStatus.inProgress, "Preparing data for training")

    sampleIdList: list[str] = []
    rowIndices: list[int] = []
    columnIndices: list[int] = []
    matrixData: list[int] = []

    outputMatrix = np.zeros((len(mappedSampleObjList), ), dtype = np.int32)

    logging.info(">> [MicrobiomeForensics] Preparing data for training. Generating input and output matrices")

    for i, sample in enumerate(mappedSampleObjList):

        sampleIdList.append(sample.sampleId)
        outputMatrix[i] = uniqueBodySite[sample.bodySite]

        # Asseble the future input matrix into three lists, with each index between them representing a data point
        for taxon in sample.taxons:
            rowIndices.append(i) # The sample index (x coordinate / row of the matrix)
            columnIndices.append(uniqueTaxons[taxon.taxonId.rstrip("\x00")]) # The feature (y coordinate / column of the matrix)
            matrixData.append(taxon.count) # The value that is represented by x and y

    inputMatrixShape = (len(mappedSampleObjList), len(uniqueTaxons))

    # Convert int32 to unsigned int16
    if quantize:
        for i, num in enumerate(matrixData):
            if num > 65535: matrixData[i] = 65535
        matrixData = np.array(matrixData).astype(np.ushort)

        # Assemble the input matrix in a sparse representation
        inputMatrix = sparse.csr_matrix((matrixData, (rowIndices, columnIndices)), inputMatrixShape, dtype = np.ushort)
    else:
        inputMatrix = sparse.csr_matrix((matrixData, (rowIndices, columnIndices)), inputMatrixShape, dtype = np.int32)

    # Remove features that carry low information
    inputMatrix, percentileModel = selectPercentile(
        inputMatrix,
        outputMatrix,
        experiment.parameters["percentile"],
        experiment.parameters["validation"],
        experiment.parameters["trainedModel"]
    )

    logging.info(">> [MicrobiomeForensics] Matrices generated")
    logging.info(f">> [MicrobiomeForensics] Input matrix shape: {inputMatrix.shape}. Output matrix shape: {outputMatrix.shape}")

    if experiment.parameters["cache"] and not experiment.parameters["validation"] and isMatrixCacheValid(cacheNameMatrix):
        experiment.updateStatus(ExperimentStatus.inProgress, "Uploading processed data to cache")
        cacheMatrix(
            cacheNameMatrix,
            inputMatrix,
            outputMatrix,
            sampleIdList,
            uniqueBodySite,
            uniqueTaxons,
            percentileModel,
            experiment.spaceId
        )

    plots(mappedSampleObjList, experiment)

    return MatrixTuple(inputMatrix, outputMatrix, sampleIdList, uniqueBodySite, uniqueTaxons)
