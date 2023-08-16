from typing import Optional
from pathlib import Path

import csv
import logging

from coretex import CustomSample, CustomDataset, Experiment, folder_manager
from coretex.bioinformatics import ctx_qiime2


FORWARD_SUMMARY_NAME = "forward-seven-number-summaries.tsv"
REVERSE_SUMMARY_NAME = "reverse-seven-number-summaries.tsv"
CASEINSENSITIVE_NAMES = ["id", "sampleid", "sample id", "sample-id", "featureid" ,"feature id", "feature-id"]
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
        raise RuntimeError(">> [Microbiome Analysis] Forward read truncLen could not be determined automatically")

    return truncLen


def columnNamePresent(metadataPath: Path, columnName: str) -> bool:
    with metadataPath.open("r") as metadata:
        for row in csv.reader(metadata, delimiter = "\t"):
            return columnName in row

    raise RuntimeError(">> [Microbiome Analysis] Metadata file is empty")


def convertMetadata(metadataPath: Path) -> Path:
    if metadataPath.suffix != ".csv" and metadataPath.suffix != ".tsv":
        raise ValueError(">> [Microbiome Analysis] Metadata has to be either tsv or csv")

    if metadataPath.suffix == ".csv":
        newMetadataPath = folder_manager.temp / f"{metadataPath.stem}.tsv"

        with metadataPath.open("r") as inputMetadata, newMetadataPath.open("w") as outputMetadata:
            outputTsv = csv.writer(outputMetadata, delimiter = "\t")

            for row in csv.reader(inputMetadata):
                outputTsv.writerow(row)

        metadataPath = newMetadataPath

    with metadataPath.open("r") as metadata:
        for row in csv.reader(metadata, delimiter = "\t"):
            break

        for columnName in row:
            if columnName.lower() in CASEINSENSITIVE_NAMES or columnName in CASESENSITIVE_NAMES:
                return metadataPath

    raise ValueError(f">> [Microbiome Analysis] Sample ID column not found. Recognized column names are: (case insensitive) - {CASEINSENSITIVE_NAMES}, (case sensitive) - {CASESENSITIVE_NAMES}")


def getMetadata(sample: CustomSample, metadataFileNme: str) -> Path:
    metadataPath = sample.joinPath(metadataFileNme)
    if metadataPath.suffix != ".tsv":
        metadataPath = metadataPath.parent / f"{metadataPath.stem}.tsv"

    return metadataPath
