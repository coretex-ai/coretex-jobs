from typing import Any, Optional
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import json

from xgboost import XGBClassifier

from coretex import folder_manager

from load_data import loadDataAtlas
from load_data_std import loadDataStd


def badRequest(error: str) -> dict[str, Any]:
    return {
        "code": 400,
        "body": {
            "error": error
        }
    }


def success(data: Optional[Any] = None) -> dict[str, Any]:
    if data is None:
        data = {}

    return {
        "code": 200,
        "body": data
    }


def prepareInputData(inputPath: Path) -> Path:
    if is_zipfile(inputPath):
        with ZipFile(inputPath, 'r') as zip_ref:
            unzipDir = folder_manager.temp / "function"

            zip_ref.extractall(unzipDir)
            if len(list(unzipDir.iterdir())) == 1:
                return list(unzipDir.iterdir())[0]

            return unzipDir

    return inputPath


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    modelDir = requestData.get("model")
    model = XGBClassifier()
    model.load_model(modelDir / "model.txt")

    with open(modelDir / "model_descriptor.json", "r") as jsonFile:
        modelDescriptor = json.load(jsonFile)

    dataFormat = int(requestData.get("dataFormat"))  # 0 - MBA, 1 - Microbiome Forensics Institute Zuric
    inputPath = requestData.get("inputFile")

    if not isinstance(inputPath, Path):
        return badRequest("Invalid input data")

    inputPath = prepareInputData(inputPath)

    if dataFormat == 0 and inputPath.is_file():
        percentile = modelDescriptor.get("percentile")

        modelInput, uniqueBodySites, sampleIdList = loadDataAtlas(inputPath, modelDir, percentile)
    elif dataFormat == 1 and inputPath.is_dir():
        level = modelDescriptor.get("taxonomicLevel")

        modelInput, uniqueBodySites, sampleIdList = loadDataStd(inputPath, modelDir, level)
    else:
        return badRequest("Invalid data format")

    predicted = model.predict(modelInput)

    predBodySites: list[str] = []
    reversedUniqueBodySites = {v: k for k, v in uniqueBodySites.items()}

    for predValue in predicted:
        predBodySites.append(reversedUniqueBodySites[predValue])

    return success({
        "name": sampleIdList,
        "bodySite": predBodySites
    })
