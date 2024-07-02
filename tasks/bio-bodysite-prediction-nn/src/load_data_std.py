from typing import Optional
from pathlib import Path

import os
import json
import glob
import logging
import pickle
import time

from coretex import TaskRun, CustomDataset, TaskRunStatus, folder_manager

from .utils import savePlotFig
from .objects import Sample, Taxon


def loadDataStd(
    taskRun: TaskRun[CustomDataset],
    dataset: CustomDataset,
    datasetPath: Path,
    level: int,
    validBodySites: Optional[dict[str, int]] = None,
    validTaxons: Optional[dict[str, int]] = None
) -> tuple[dict[str, int], dict[str, int], int]:

    logging.info(">> [MicrobiomeForensics] Downloading dataset...")
    taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset...")
    dataset.download()

    taskRun.updateStatus(TaskRunStatus.inProgress, "Loading data")

    taxonDistributionSavePath = folder_manager.temp / "taxon_histogram.png"
    classDistributionSavePath = folder_manager.temp / "body_site_histogram.png"

    datasetLen = len(taskRun.dataset.samples)
    if datasetLen < 10:
        raise RuntimeError(f">> [BioInformatics] You have insufficient number of samples in your dataset ({datasetLen})")

    uniqueTaxons: dict[str, int] = {}
    uniqueBodySites: dict[str, int] = {}

    taxonDistribution: dict[str, int] = {}
    classDistribution: dict[str, int] = {}

    logging.info(f">> [MicrobiomeForensics] Loading dataset")
    startTime = time.time()

    for sample in dataset.samples:
        sample.unzip()
        samplePath = glob.glob(os.path.join(sample.path, f"*.json"))[0]

        with open(samplePath, "r") as f:
            sampleDict = json.load(f)

        if validBodySites is not None and sampleDict["body_site"] not in validBodySites:
            continue

        sampleObj = Sample(sampleDict["_id"]["$oid"], sampleDict["body_site"], None, [])

        if not sampleDict["body_site"] in classDistribution:
            classDistribution[sampleDict["body_site"]] = 1
        else:
            classDistribution[sampleDict["body_site"]] += 1

        taxons = loadTaxons(sampleDict, level)

        if validTaxons is not None and any(taxon not in validTaxons for taxon in taxons.keys()):
            continue

        for taxonId, taxonCount in zip(taxons.keys(), taxons.values()):
            sampleObj.addTaxon(Taxon(taxonId, taxonCount))

            if taxonId not in taxonDistribution:
                taxonDistribution[taxonId] = taxonCount
            else:
                taxonDistribution[taxonId] += taxonCount

        with datasetPath.joinpath(sampleObj.sampleId).open("wb") as file:
            pickle.dump(sampleObj, file)

    if validBodySites is not None and validTaxons is not None:
        uniqueBodySites = validBodySites
        uniqueTaxons = validTaxons
    else:
        for bodySite in classDistribution:
            uniqueBodySites[bodySite] = len(uniqueBodySites)

        for taxon in taxonDistribution:
            uniqueTaxons[taxon] = len(uniqueTaxons)

    logging.info(f">> [MicrobiomeForensics] Loaded data in: {time.time() - startTime:.1f}s")

    taxonDistributionValues = list(taxonDistribution.values())
    taxonDistribution = {k:v for k, v in taxonDistribution.items() if v > (max(taxonDistributionValues)/10)}

    savePlotFig(
        taskRun,
        classDistribution,
        classDistributionSavePath,
        "body_site_histogram.png",
        True,
        "Body Site",
        "Number of samples",
        "Class distribution"
    )

    savePlotFig(
        taskRun,
        taxonDistribution,
        taxonDistributionSavePath,
        "taxon_histogram.png",
        True,
        "Taxon",
        "Total count",
        "Representation of dataset sample"
    )

    return uniqueBodySites, uniqueTaxons, datasetLen


def loadTaxons(sample: dict, level: int) -> dict[str, int]:
    taxons: dict[str, int] = {}
    for bacteria in sample["97"]:
        taxon = bacteria["taxon"].split(";")

        if len(taxon) <= level:
            logging.warning(f">> [MicrobiomeForensics] Sample: {sample['_id']['$oid']} does not have taxonomic level {level} and will be skipped")
            continue

        taxons[taxon[level]] = bacteria["count"]

    return taxons