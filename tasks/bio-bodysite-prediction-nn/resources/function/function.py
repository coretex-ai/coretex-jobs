from typing import Any
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import json

from coretex import folder_manager, functions

import numpy as np

from load_data import loadDataAtlas
from load_data_std import loadDataStd

from model import Model
from dataset import loadDataset


def unzip(inputPath: Path, dataFormat: int) -> Path:
    if is_zipfile(inputPath):
        with ZipFile(inputPath, 'r') as zipFile:
            unzipDir = folder_manager.createTempFolder("function")

            zipFile.extractall(unzipDir)
            if dataFormat != 0:
                return unzipDir

            for element in unzipDir.iterdir():
                if element.name != "__MACOSX":
                    return element

    return inputPath


def inference(modelInput: Path, model: Model, uniqueTaxons: dict[str, int]) -> np.ndarray:
    BATCHE_SIZE = 562
    sampleCount = len(list(modelInput.iterdir()))

    dataset = loadDataset(modelInput, uniqueTaxons)
    data = dataset.batch(BATCHE_SIZE)

    return model.predict(data, sampleCount)


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    modelDir = Path.cwd().parent

    with open(modelDir / "model_descriptor.json", "r") as jsonFile:
        modelDescriptor = json.load(jsonFile)

    dataFormatRaw = requestData.get("dataFormat")
    if not isinstance(dataFormatRaw, str) and not isinstance(dataFormatRaw, int):
        return functions.badRequest("Invalid dataFormat. (0 - MBA, 1 - Microbiome Forensics Institute Zuric)")

    dataFormat = int(dataFormatRaw)  # 0 - MBA, 1 - Microbiome Forensics Institute Zuric
    inputPath = requestData.get("inputFile")

    if not isinstance(inputPath, Path):
        return functions.badRequest("Invalid input data")

    inputPath = unzip(inputPath, dataFormat)

    if dataFormat == 0 and inputPath.is_file():
        modelInput, uniqueTaxons, uniqueBodySites, sampleIdList = loadDataAtlas(inputPath, modelDir)
    elif dataFormat == 1 and inputPath.is_dir():
        level = modelDescriptor.get("taxonomicLevel")

        modelInput, uniqueTaxons, uniqueBodySites, sampleIdList = loadDataStd(inputPath, modelDir, level)
    else:
        return functions.badRequest("Invalid data format")

    model = Model.load(modelDir / "model")

    predicted = inference(modelInput, model, uniqueTaxons)

    predBodySites: list[str] = []
    reversedUniqueBodySites = {v: k for k, v in uniqueBodySites.items()}

    for predValue in predicted:
        predBodySites.append(reversedUniqueBodySites[predValue])

    return functions.success({
        "name": sampleIdList,
        "bodySite": predBodySites
    })
