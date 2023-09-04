import logging

from coretex import CustomDataset, Run
from coretex.utils import hashCacheName


def getCacheNameOne(run: Run) -> str:
    if run.parameters["barcodeColumn"]:
        prefix = f"{run.id} - Step 1: Multiplexed"
    else:
        prefix = f"{run.id} - Step 1: Demultiplexed"

    paramList = [
        "Step 1",
        str(run.dataset.id),
        str(run.parameters["metadataFileName"]),
        str(run.parameters["forwardAdapter"]),
        str(run.parameters["reverseAdapter"])
    ]

    if not run.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameTwo(run: Run) -> str:
    prefix = f"{run.id} - Step 2: Demultiplexing"

    paramList = [
        "Step 2",
        str(run.dataset.id),
        str(run.parameters["metadataFileName"]),
        str(run.parameters["barcodeColumn"]),
        str(run.parameters["forwardAdapter"]),
        str(run.parameters["reverseAdapter"])
    ]

    if not run.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameThree(run: Run) -> str:
    prefix = f"{run.id} - Step 3: DADA2"

    paramList = [
        "Step 3",
        str(run.dataset.id),
        str(run.parameters["metadataFileName"]),
        str(run.parameters["barcodeColumn"]),
        str(run.parameters["forwardAdapter"]),
        str(run.parameters["reverseAdapter"]),
        str(run.parameters["trimLeftF"]),
        str(run.parameters["trimLeftR"]),
        str(run.parameters["truncLenF"]),
        str(run.parameters["truncLenR"])
    ]

    if not run.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameFour(run: Run) -> str:
    prefix = f"{run.id} - Step 4: Clustering - {run.parameters['clusteringMethod']}"

    paramList = [
        "Step 4",
        str(run.dataset.id),
        str(run.parameters["metadataFileName"]),
        str(run.parameters["barcodeColumn"]),
        str(run.parameters["forwardAdapter"]),
        str(run.parameters["reverseAdapter"]),
        str(run.parameters["trimLeftF"]),
        str(run.parameters["trimLeftR"]),
        str(run.parameters["truncLenF"]),
        str(run.parameters["truncLenR"]),
        str(run.parameters["clusteringMethod"]),
        str(run.parameters["referenceDataset"]),
        str(run.parameters["percentIdentity"])
    ]

    if not run.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameFive(run: Run) -> str:
    prefix = f"{run.id} - Step 5: Taxonomic analysis"

    paramList = [
        "Step 5",
        str(run.dataset.id),
        str(run.parameters["metadataFileName"]),
        str(run.parameters["barcodeColumn"]),
        str(run.parameters["forwardAdapter"]),
        str(run.parameters["reverseAdapter"]),
        str(run.parameters["trimLeftF"]),
        str(run.parameters["trimLeftR"]),
        str(run.parameters["truncLenF"]),
        str(run.parameters["truncLenR"]),
        str(run.parameters["classifier"])
    ]

    if not run.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameSix(run: Run) -> str:
    prefix = f"{run.id} - Step 6: Phylogenetic tree"

    paramList = [
        "Step 6",
        str(run.dataset.id),
        str(run.parameters["metadataFileName"]),
        str(run.parameters["barcodeColumn"]),
        str(run.parameters["forwardAdapter"]),
        str(run.parameters["reverseAdapter"]),
        str(run.parameters["trimLeftF"]),
        str(run.parameters["trimLeftR"]),
        str(run.parameters["truncLenF"]),
        str(run.parameters["truncLenR"])
    ]

    if not run.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameSeven(run: Run) -> str:
    prefix = f"{run.id} - Step 7: Alpha & Beta diversity"

    paramList = [
        "Step 7",
        str(run.dataset.id),
        str(run.parameters["metadataFileName"]),
        str(run.parameters["barcodeColumn"]),
        str(run.parameters["forwardAdapter"]),
        str(run.parameters["reverseAdapter"]),
        str(run.parameters["trimLeftF"]),
        str(run.parameters["trimLeftR"]),
        str(run.parameters["truncLenF"]),
        str(run.parameters["truncLenR"]),
        str(run.parameters["samplingDepth"]),
        str(run.parameters["maxDepth"]),
        str(run.parameters["targetTypeColumn"])
    ]

    if not run.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCache(cacheName: str, run: Run) -> CustomDataset:
    logging.info(f">> [Microbiome analysis] Searching for cache {cacheName}")

    cacheHash = cacheName.split("_")[1]
    caches = CustomDataset.fetchAll(queryParameters = [f"name={cacheHash}", "include_sessions=1"])

    for cache in caches:
        if cache.count != 0:
            logging.info(">> [Microbiome analysis] Cache found!")
            cache.download()
            uploadCacheAsArtifact(cache, run)

            return cache

    raise ValueError(">> [Microbiome analysis] Cache does not exist!")


def uploadCacheAsArtifact(cache: CustomDataset, run: Run) -> None:
    stepName = cache.name.split(" - ")[1].split("_")[0]
    for sample in cache.samples:
        sampleName = sample.name.split('_')[0]
        run.createQiimeArtifact(f"{stepName}/{sampleName}", sample.zipPath)


def cacheExists(cacheName: str) -> bool:
    cacheHash = cacheName.split("_")[1]
    caches = CustomDataset.fetchAll(queryParameters = [f"name={cacheHash}", "include_sessions=1"])

    for cache in caches:
        if cache.count != 0:
            return True

    return False
