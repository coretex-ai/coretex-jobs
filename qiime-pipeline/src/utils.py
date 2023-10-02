from typing import Optional
from pathlib import Path

import csv
import logging

import cchardet
import pandas as pd

from coretex import CustomSample, folder_manager
from coretex.bioinformatics import ctx_qiime2


FORWARD_SUMMARY_NAME = "forward-seven-number-summaries.tsv"
REVERSE_SUMMARY_NAME = "reverse-seven-number-summaries.tsv"
CASEINSENSITIVE_NAMES = ["id", "sampleid", "sample id", "sample-id", "featureid" ,"feature id", "feature-id", "sample_id", "sample.id"]
CASESENSITIVE_NAMES = ["#SampleID" , "#Sample ID", "#OTUID", "#OTU ID", "sample_name"]


def demuxSummarize(sample: CustomSample, outputDir: Path) -> Path:
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
        strand = "Forward" if forward else "Reverse"
        raise RuntimeError(f">> [Qiime: DADA2] {strand} read truncLen could not be determined automatically")

    return truncLen


def columnNamePresent(metadataPath: Path, columnName: str) -> bool:
    with metadataPath.open("r") as metadata:
        for row in csv.reader(metadata, delimiter = "\t"):
            return columnName in row

    raise RuntimeError(">> [Qiime: Import] Metadata file is empty")


def detectFileEncoding(path: Path) -> Optional[str]:
    if path.stat().st_size < 10:
        raise ValueError(">> [Qiime: Import] Metadate file is too small")

    with path.open("rb") as file:
        encoding = cchardet.detect(file.read())["encoding"]

    if encoding is None:
        logging.warning(">> [Qiime: Import] Could not determine metadata encoding")

    return encoding


def convertMetadata(metadataPath: Path) -> Path:
    newMetadataPath = folder_manager.temp / f"{metadataPath.stem}.tsv"
    if metadataPath.suffix != ".csv" and metadataPath.suffix != ".tsv":
        raise ValueError(">> [Qiime: Import] Metadata has to be either tsv or csv")

    if metadataPath.suffix == ".csv":
        metadata = pd.read_csv(metadataPath, encoding = detectFileEncoding(metadataPath))
    else:
        metadata = pd.read_csv(metadataPath, encoding = detectFileEncoding(metadataPath), delimiter = "\t")

    for columnName in metadata.columns:
        if columnName.lower() in CASEINSENSITIVE_NAMES or columnName in CASESENSITIVE_NAMES:
            sampleIdColumn = metadata.pop(columnName)
            metadata.insert(0, "sampleId", sampleIdColumn)
            break

    if metadata.columns[0] != "sampleId":
        raise ValueError(f">> [Qiime: Import] Sample ID column not found. Recognized column names are: (case insensitive) - {CASEINSENSITIVE_NAMES}, (case sensitive) - {CASESENSITIVE_NAMES}")

    for sampleId in metadata["sampleId"]:
        sampleIdSplit = str(sampleId).split("_")
        if len(sampleIdSplit) > 1:
            metadata["sampleId"].replace(sampleId, sampleIdSplit[0], inplace = True)

    metadata.to_csv(newMetadataPath, "\t", index = False)

    return newMetadataPath
