from pathlib import Path

import json
import pickle
import logging

from coretex import folder_manager

from objects import Sample, Taxon


def loadDataStd(inputPath: Path, modelDir: Path, level: int) -> tuple[Path, dict[str, int], dict[str, int], list[str]]:
    with open(modelDir / "uniqueTaxons.pkl", "rb") as f:
        uniqueTaxons: dict[str, int] = pickle.load(f)

    with open(modelDir / "uniqueBodySites.pkl", "rb") as f:
        uniqueBodySites: dict[str, int] = pickle.load(f)

    datasetPath = folder_manager.createTempFolder("dataset")

    for samplePath in inputPath.iterdir():
        if samplePath.suffix != ".json":
            continue

        with open(samplePath, "r") as f:
            sample = json.load(f)

        sampleObj = Sample(sample["_id"]["$oid"])

        taxons = loadTaxons(sample, level)
        for taxonId, taxonCount in taxons.items():
            if taxonId in uniqueTaxons:
                sampleObj.addTaxon(Taxon(taxonId, taxonCount))

        with datasetPath.joinpath(sampleObj.sampleId).open("wb") as file:
            pickle.dump(sampleObj, file)

    sampleIdList: list[str] = []
    for path in datasetPath.iterdir():
        sampleIdList.append(path.name)

    return datasetPath, uniqueTaxons, uniqueBodySites, sampleIdList


def loadTaxons(sample: dict, level: int) -> dict[str, int]:
    taxons: dict[str, int] = {}
    for bacteria in sample["97"]:
        taxon = bacteria["taxon"].split(";")

        if len(taxon) <= level:
            logging.warning(f">> [MicrobiomeForensics] Sample: {sample['_id']['$oid']} does not have taxonomic level {level} and will be skipped")
            continue

        taxons[taxon[level]] = bacteria["count"]

    return taxons
