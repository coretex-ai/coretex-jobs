from pathlib import Path

import json
import pickle

import numpy as np


def loadDataStd(inputPath: Path, modelDir: Path, level: int) -> tuple[np.ndarray, dict[str, int], list[str]]:
    with open(modelDir / "uniqueTaxons.pkl", "rb") as f:
        uniqueTaxons = pickle.load(f)

    with open(modelDir / "uniqueBodySites.pkl", "rb") as f:
        uniqueBodySites = pickle.load(f)

    inputMatrix = np.zeros((len(list(inputPath.iterdir())), len(uniqueTaxons)))
    sampleIdList = []

    for i, samplePath in enumerate(inputPath.iterdir()):
        if samplePath.suffix != ".json":
            continue

        with open(samplePath, "r") as f:
            sample = json.load(f)

        sampleIdList.append(sample["_id"]["$oid"])
        for bacteria in sample["97"]:
            taxons = bacteria["taxon"].split(";")
            taxon = taxons[level]

            if taxon not in uniqueTaxons:
                continue

            encodedTaxon = uniqueTaxons[taxon]

            c = bacteria["count"]
            inputMatrix[i, encodedTaxon] += c

    return inputMatrix, uniqueBodySites, sampleIdList
