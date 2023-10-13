import logging

from coretex import CustomDataset, TaskRun
from coretex.utils import hashCacheName


def getCacheNameOne(taskRun: TaskRun) -> str:
    if taskRun.parameters["barcodeColumn"]:
        prefix = f"{taskRun.id} - Step 1: Multiplexed"
    else:
        prefix = f"{taskRun.id} - Step 1: Demultiplexed"

    paramList = [
        "Step 1",
        str(taskRun.dataset.id),
        str(taskRun.parameters["metadataFileName"]),
        str(taskRun.parameters["forwardAdapter"]),
        str(taskRun.parameters["reverseAdapter"])
    ]

    if not taskRun.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameTwo(taskRun: TaskRun) -> str:
    prefix = f"{taskRun.id} - Step 2: Demultiplexing"

    paramList = [
        "Step 2",
        str(taskRun.dataset.id),
        str(taskRun.parameters["metadataFileName"]),
        str(taskRun.parameters["barcodeColumn"]),
        str(taskRun.parameters["forwardAdapter"]),
        str(taskRun.parameters["reverseAdapter"])
    ]

    if not taskRun.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameThree(taskRun: TaskRun) -> str:
    prefix = f"{taskRun.id} - Step 3: DADA2"

    paramList = [
        "Step 3",
        str(taskRun.dataset.id),
        str(taskRun.parameters["metadataFileName"]),
        str(taskRun.parameters["barcodeColumn"]),
        str(taskRun.parameters["forwardAdapter"]),
        str(taskRun.parameters["reverseAdapter"]),
        str(taskRun.parameters["trimLeftF"]),
        str(taskRun.parameters["trimLeftR"]),
        str(taskRun.parameters["truncLenF"]),
        str(taskRun.parameters["truncLenR"])
    ]

    if not taskRun.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameFour(taskRun: TaskRun) -> str:
    prefix = f"{taskRun.id} - Step 4: Clustering - {taskRun.parameters['clusteringMethod']}"

    paramList = [
        "Step 4",
        str(taskRun.dataset.id),
        str(taskRun.parameters["metadataFileName"]),
        str(taskRun.parameters["barcodeColumn"]),
        str(taskRun.parameters["forwardAdapter"]),
        str(taskRun.parameters["reverseAdapter"]),
        str(taskRun.parameters["trimLeftF"]),
        str(taskRun.parameters["trimLeftR"]),
        str(taskRun.parameters["truncLenF"]),
        str(taskRun.parameters["truncLenR"]),
        str(taskRun.parameters["clusteringMethod"]),
        str(taskRun.parameters["referenceDataset"]),
        str(taskRun.parameters["percentIdentity"])
    ]

    if not taskRun.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameFive(taskRun: TaskRun) -> str:
    prefix = f"{taskRun.id} - Step 5: Taxonomic analysis"

    paramList = [
        "Step 5",
        str(taskRun.dataset.id),
        str(taskRun.parameters["metadataFileName"]),
        str(taskRun.parameters["barcodeColumn"]),
        str(taskRun.parameters["forwardAdapter"]),
        str(taskRun.parameters["reverseAdapter"]),
        str(taskRun.parameters["trimLeftF"]),
        str(taskRun.parameters["trimLeftR"]),
        str(taskRun.parameters["truncLenF"]),
        str(taskRun.parameters["truncLenR"]),
        str(taskRun.parameters["classifier"])
    ]

    if not taskRun.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameSix(taskRun: TaskRun) -> str:
    prefix = f"{taskRun.id} - Step 6: Phylogenetic tree"

    paramList = [
        "Step 6",
        str(taskRun.dataset.id),
        str(taskRun.parameters["metadataFileName"]),
        str(taskRun.parameters["barcodeColumn"]),
        str(taskRun.parameters["forwardAdapter"]),
        str(taskRun.parameters["reverseAdapter"]),
        str(taskRun.parameters["trimLeftF"]),
        str(taskRun.parameters["trimLeftR"]),
        str(taskRun.parameters["truncLenF"]),
        str(taskRun.parameters["truncLenR"])
    ]

    if not taskRun.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCacheNameSeven(taskRun: TaskRun) -> str:
    prefix = f"{taskRun.id} - Step 7: Alpha & Beta diversity"

    paramList = [
        "Step 7",
        str(taskRun.dataset.id),
        str(taskRun.parameters["metadataFileName"]),
        str(taskRun.parameters["barcodeColumn"]),
        str(taskRun.parameters["forwardAdapter"]),
        str(taskRun.parameters["reverseAdapter"]),
        str(taskRun.parameters["trimLeftF"]),
        str(taskRun.parameters["trimLeftR"]),
        str(taskRun.parameters["truncLenF"]),
        str(taskRun.parameters["truncLenR"]),
        str(taskRun.parameters["samplingDepth"]),
        str(taskRun.parameters["maxDepth"]),
        str(taskRun.parameters["targetTypeColumn"])
    ]

    if not taskRun.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList)).replace("+", "0")


def getCache(cacheName: str, taskRun: TaskRun) -> CustomDataset:
    logging.info(f">> [Microbiome analysis] Searching for cache {cacheName}")

    cacheHash = cacheName.split("_")[1]
    caches = CustomDataset.fetchAll(
        name = cacheHash,
        include_sessions = 1
    )

    for cache in caches:
        if cache.count != 0:
            logging.info(">> [Microbiome analysis] Cache found!")
            cache.download()
            uploadCacheAsArtifact(cache, taskRun)

            return cache

    raise ValueError(">> [Microbiome analysis] Cache does not exist!")


def uploadCacheAsArtifact(cache: CustomDataset, taskRun: TaskRun) -> None:
    stepName = cache.name.split(" - ")[1].split("_")[0]
    for sample in cache.samples:
        sampleName = sample.name.split('_')[0]
        taskRun.createQiimeArtifact(f"{stepName}/{sampleName}", sample.zipPath)


def cacheExists(cacheName: str) -> bool:
    cacheHash = cacheName.split("_")[1]
    caches = CustomDataset.fetchAll(
        name = cacheHash,
        include_sessions = 1
    )

    for cache in caches:
        if cache.count != 0:
            return True

    return False
