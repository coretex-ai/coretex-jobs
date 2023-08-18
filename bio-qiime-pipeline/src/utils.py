from typing import Optional, Tuple, Dict
from pathlib import Path

import logging
import csv
import pandas as pd

from coretex import CustomSample, CustomDataset, folder_manager, Experiment
from coretex.bioinformatics import ctx_qiime2
from coretex.utils.hash import hashCacheName

import chardet


FORWARD_SUMMARY_NAME = "forward-seven-number-summaries.tsv"
REVERSE_SUMMARY_NAME = "reverse-seven-number-summaries.tsv"
CASEINSENSITIVE_NAMES = ["id", "sampleid", "sample id", "sample-id", "featureid" ,"feature id", "feature-id", "sample_id", "sample.id"]
CASESENSITIVE_NAMES = ["#SampleID" , "#Sample ID", "#OTUID", "#OTU ID", "sample_name"]


def summarizeSample(sample: CustomSample, outputDir: Path) -> Path:
    demuxPath = sample.path / "demux.qza"
    visualizationPath = outputDir / "demux.qzv"

    ctx_qiime2.demuxSummarize(str(demuxPath), str(visualizationPath))
    return visualizationPath


def determineTruncLen(sample: CustomSample, forward: bool) -> int:
    sample.unzip()

    summariesFileName = FORWARD_SUMMARY_NAME if forward else REVERSE_SUMMARY_NAME
    summariesTsv = list(sample.path.rglob(summariesFileName))[0]

    truncLen: Optional[int] = None
    with summariesTsv.open("r") as file:
        summaries = list(csv.reader(file, delimiter = "\t"))

    # summaries will allways have the median quality at row 5 of the csv
    medianQualitiesStr = summaries[5]
    medianQualitiesStr.pop(0)  # The first value will be "50%" and not a quality score
    medianQualities = [float(x) for x in medianQualitiesStr]

    highestScore = max(medianQualities)
    for index, qualityScore in enumerate(medianQualities):
        if qualityScore < highestScore * 0.7:
            truncLen = index
            break

    if not truncLen:
        raise RuntimeError(">> [Microbiome analysis] Forward read truncLen could not be determined automatically")

    return truncLen


def loadSingleEnd(sample: CustomSample) -> Tuple[Path, str]:
    sample.unzip()

    filePathList = list(sample.path.glob("*.fastq*"))
    if len(filePathList) == 1:
        filePath = filePathList[0]
        return filePath, filePath.name.split("_")[0]

    raise ValueError(f">> [Microbiome analysis] Sample \"{sample.name}\" must contain exactly one fastq file")


def loadPairedEnd(sample: CustomSample) -> Tuple[Path, Path, str]:
    sample.unzip()

    forwardPathList = list(sample.path.glob("*_R1_*.fastq*"))
    reversePathList = list(sample.path.glob("*_R2_*.fastq*"))

    if len(forwardPathList) != 1 or len(reversePathList) != 1:
        raise ValueError(f">> [Microbiome analysis] Invalid paired-end sample: {sample.name}. Must contain 2 files, one with \"_R1_\" and another with \"_R2_\" in name")

    forwardPath = forwardPathList[0]
    reversePath = reversePathList[0]

    return forwardPath, reversePath, forwardPath.name.split("_")[0]


def isPairedEnd(dataset: CustomDataset) -> bool:
    for sample in dataset.samples:
        sample.unzip()

        if sample.name.startswith("_metadata"):
            continue

        if len(list(sample.path.glob("*.fastq*"))) != 2:
            return False

    return True


def isGzCompressed(dataset: CustomDataset) -> bool:
    for sample in dataset.samples:
        sample.unzip()
        if len(list(sample.path.glob("*.fastq.gz"))) > 0:
            return True

    return False


def columnNamePresent(metadataPath: Path, columnName: str) -> bool:
    with metadataPath.open("r") as metadata:
        for row in csv.reader(metadata, delimiter = "\t"):
            return columnName in row

    raise RuntimeError(">> [Microbiome analysis] Metadata file is empty")


def detectFileEncoding(path: Path) -> Optional[str]:
    with path.open("rb") as file:
        return chardet.detect(file.read(10))["encoding"]


def convertMetadata(metadataPath: Path) -> Path:
    newMetadataPath = folder_manager.temp / f"{metadataPath.stem}.tsv"
    if metadataPath.suffix != ".csv" and metadataPath.suffix != ".tsv":
        raise ValueError(">> [Microbiome analysis] Metadata has to be either tsv or csv")

    if metadataPath.suffix == ".csv":
        metadata = pd.read_csv(metadataPath, encoding = detectFileEncoding(metadataPath))
    else:
        metadata = pd.read_csv(metadataPath, encoding = detectFileEncoding(metadataPath), delimiter = "\t")

    for i, columnName in enumerate(metadata.columns):
        if columnName.lower() in CASEINSENSITIVE_NAMES or columnName in CASESENSITIVE_NAMES:
            break

        raise ValueError(f">> [Microbiome analysis] Sample ID column not found. Recognized column names are: (case insensitive) - {CASEINSENSITIVE_NAMES}, (case sensitive) - {CASESENSITIVE_NAMES}")

    metadata.columns.values[i] = "sampleid"
    for sampleId in metadata["sampleid"]:
        sampleIdSplit = str(sampleId).split("_")
        if len(sampleIdSplit) > 1:
            metadata["sampleid"].replace(sampleId, sampleIdSplit[0], inplace = True)

    metadata.to_csv(newMetadataPath, "\t", index = False)

    return newMetadataPath


def getMetadata(sample: CustomSample, metadataFileNme: str) -> Path:
    metadataPath = sample.joinPath(metadataFileNme)
    if metadataPath.suffix != ".tsv":
        metadataPath = metadataPath.parent / f"{metadataPath.stem}.tsv"

    return metadataPath


def getDatasetName(experiment: Experiment[CustomDataset], step: int) -> str:
    if step < 1 or step > 5:
        raise ValueError(">> [Microbiome analysis] Step number has to be between 1 and 5")

    if experiment.parameters["barcodeColumn"]:
        prefix = f"{experiment.id} - Step 1: Demux"
    else:
        prefix = f"{experiment.id} - Step 1: Import"

    paramList = [
        str(experiment.dataset.id),
        str(experiment.parameters["metadataFileName"]),
        str(experiment.parameters["barcodeColumn"]),
        str(experiment.parameters["forwardAdapter"]),
        str(experiment.parameters["reverseAdapter"])
    ]

    if step > 1:
        prefix = f"{experiment.id} - Step 2: Denoise"
        paramList.extend([
            str(experiment.parameters["trimLeftF"]),
            str(experiment.parameters["trimLeftR"]),
            str(experiment.parameters["truncLenF"]),
            str(experiment.parameters["truncLenR"]),
        ])

    if step == 3:
        prefix = f"{experiment.id} - Step 3: Phylogenetic tree"
        paramList.append("Step 3")

    if step == 4:
        prefix = f"{experiment.id} - Step 4: Alpha & Beta diversity"
        paramList.extend([
            str(experiment.parameters["samplingDepth"]),
            str(experiment.parameters["maxDepth"]),
            str(experiment.parameters["targetTypeColumn"])
        ])

    if step == 5:
        prefix = f"{experiment.id} - Step 5: Taxonomic analysis"
        paramList.append(experiment.parameters["classifier"])

    if not experiment.parameters["useCache"]:
        return prefix

    return hashCacheName(prefix, "_".join(paramList))


def getCaches(experiment: Experiment[CustomDataset]) -> Dict[int, CustomDataset]:
    logging.info(f">> [Microbiome analysis] Searching for cache")

    cacheDict: Dict[int, CustomDataset] = {}
    for step in range(1, 6):
        cacheHash = getDatasetName(experiment, step).split("_")[1]
        caches = CustomDataset.fetchAll(queryParameters = [f"name={cacheHash}", "include_sessions=1"])

        if len(caches) == 0:
            continue

        for cache in caches:
            if cache.count != 0:
                dataset = CustomDataset.fetchById(cache.id)
                break

        stepName = dataset.name.split(" - ")[1].split("_")[0]
        logging.info(f">> [Microbiome analysis] Found {stepName} cache. Dataset ID: {dataset.id}")
        dataset.download()

        cacheDict[step] = dataset
        for sample in dataset.samples:
            sampleName = sample.name.split('_')[0]
            experiment.createQiimeArtifact(f"{stepName}/{sampleName}", sample.zipPath)

    return cacheDict
