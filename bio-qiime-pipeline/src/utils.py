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
