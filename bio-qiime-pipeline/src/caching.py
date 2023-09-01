import logging

from coretex import CustomDataset, Experiment
from coretex.utils import hashCacheName


def getCacheNameOne(experiment: Experiment) -> str:
    if experiment.parameters["barcodeColumn"]:
        prefix = f"{experiment.id} - Step 1: Multiplexed"
    else:
        prefix = f"{experiment.id} - Step 1: Demultiplexed"

    paramList = [
        "Step 1",
        str(experiment.dataset.id),
        str(experiment.parameters["metadataFileName"]),
        str(experiment.parameters["forwardAdapter"]),
        str(experiment.parameters["reverseAdapter"])
    ]

    if not experiment.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameTwo(experiment: Experiment) -> str:
    prefix = f"{experiment.id} - Step 2: Demultiplexing"

    paramList = [
        "Step 2",
        str(experiment.dataset.id),
        str(experiment.parameters["metadataFileName"]),
        str(experiment.parameters["barcodeColumn"]),
        str(experiment.parameters["forwardAdapter"]),
        str(experiment.parameters["reverseAdapter"])
    ]

    if not experiment.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameThree(experiment: Experiment) -> str:
    prefix = f"{experiment.id} - Step 3: DADA2"

    paramList = [
        "Step 3",
        str(experiment.dataset.id),
        str(experiment.parameters["metadataFileName"]),
        str(experiment.parameters["barcodeColumn"]),
        str(experiment.parameters["forwardAdapter"]),
        str(experiment.parameters["reverseAdapter"]),
        str(experiment.parameters["trimLeftF"]),
        str(experiment.parameters["trimLeftR"]),
        str(experiment.parameters["truncLenF"]),
        str(experiment.parameters["truncLenR"])
    ]

    if not experiment.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameFour(experiment: Experiment) -> str:
    prefix = f"{experiment.id} - Step 4: Clustering - {experiment.parameters['clusteringMethod']}"

    paramList = [
        "Step 4",
        str(experiment.dataset.id),
        str(experiment.parameters["metadataFileName"]),
        str(experiment.parameters["barcodeColumn"]),
        str(experiment.parameters["forwardAdapter"]),
        str(experiment.parameters["reverseAdapter"]),
        str(experiment.parameters["trimLeftF"]),
        str(experiment.parameters["trimLeftR"]),
        str(experiment.parameters["truncLenF"]),
        str(experiment.parameters["truncLenR"]),
        str(experiment.parameters["clusteringMethod"]),
        str(experiment.parameters["referenceDataset"]),
        str(experiment.parameters["percentIdentity"])
    ]

    if not experiment.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameFive(experiment: Experiment) -> str:
    prefix = f"{experiment.id} - Step 5: Taxonomic analysis"

    paramList = [
        "Step 5",
        str(experiment.dataset.id),
        str(experiment.parameters["metadataFileName"]),
        str(experiment.parameters["barcodeColumn"]),
        str(experiment.parameters["forwardAdapter"]),
        str(experiment.parameters["reverseAdapter"]),
        str(experiment.parameters["trimLeftF"]),
        str(experiment.parameters["trimLeftR"]),
        str(experiment.parameters["truncLenF"]),
        str(experiment.parameters["truncLenR"]),
        str(experiment.parameters["classifier"])
    ]

    if not experiment.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameSix(experiment: Experiment) -> str:
    prefix = f"{experiment.id} - Step 6: Phylogenetic tree"

    paramList = [
        "Step 6",
        str(experiment.dataset.id),
        str(experiment.parameters["metadataFileName"]),
        str(experiment.parameters["barcodeColumn"]),
        str(experiment.parameters["forwardAdapter"]),
        str(experiment.parameters["reverseAdapter"]),
        str(experiment.parameters["trimLeftF"]),
        str(experiment.parameters["trimLeftR"]),
        str(experiment.parameters["truncLenF"]),
        str(experiment.parameters["truncLenR"])
    ]

    if not experiment.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameSeven(experiment: Experiment) -> str:
    prefix = f"{experiment.id} - Step 7: Alpha & Beta diversity"

    paramList = [
        "Step 7",
        str(experiment.dataset.id),
        str(experiment.parameters["metadataFileName"]),
        str(experiment.parameters["barcodeColumn"]),
        str(experiment.parameters["forwardAdapter"]),
        str(experiment.parameters["reverseAdapter"]),
        str(experiment.parameters["trimLeftF"]),
        str(experiment.parameters["trimLeftR"]),
        str(experiment.parameters["truncLenF"]),
        str(experiment.parameters["truncLenR"]),
        str(experiment.parameters["samplingDepth"]),
        str(experiment.parameters["maxDepth"]),
        str(experiment.parameters["targetTypeColumn"])
    ]

    if not experiment.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCache(cacheName: str, experiment: Experiment) -> CustomDataset:
    logging.info(f">> [Microbiome analysis] Searching for cache {cacheName}")

    cacheHash = cacheName.split("_")[1]
    caches = CustomDataset.fetchAll(queryParameters = [f"name={cacheHash}", "include_sessions=1"])

    for cache in caches:
        if cache.count != 0:
            logging.info(">> [Microbiome analysis] Cache found!")
            cache.download()
            uploadCacheAsArtifact(cache, experiment)

            return cache

    raise ValueError(">> [Microbiome analysis] Cache does not exist!")


def uploadCacheAsArtifact(cache: CustomDataset, experiment: Experiment) -> None:
    stepName = cache.name.split(" - ")[1].split("_")[0]
    for sample in cache.samples:
        sampleName = sample.name.split('_')[0]
        experiment.createQiimeArtifact(f"{stepName}/{sampleName}", sample.zipPath)


def cacheExists(cacheName: str) -> bool:
    cacheHash = cacheName.split("_")[1]
    caches = CustomDataset.fetchAll(queryParameters = [f"name={cacheHash}", "include_sessions=1"])

    for cache in caches:
        if cache.count != 0:
            return True

    return False
