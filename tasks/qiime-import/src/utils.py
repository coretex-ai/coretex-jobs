from typing import Optional
from pathlib import Path

import csv
import logging

from coretex import folder_manager

import cchardet
import pandas as pd


CASEINSENSITIVE_NAMES = ["id", "sampleid", "sample id", "sample-id", "featureid" ,"feature id", "feature-id", "sample_id", "sample.id"]
CASESENSITIVE_NAMES = ["#SampleID" , "#Sample ID", "#OTUID", "#OTU ID", "sample_name"]


def columnNamePresent(metadataPath: Path, columnName: str) -> bool:
    with metadataPath.open("r") as metadata:
        for row in csv.reader(metadata, delimiter = "\t"):
            return columnName in row

    raise RuntimeError(">> [Qiime: Import] Metadata file is empty")


def detectFileEncoding(path: Path) -> Optional[str]:
    if path.stat().st_size < 10:
        raise ValueError(">> [Qiime: Import] Metadate file is too small")

    with path.open("rb") as file:
        encoding: Optional[str] = cchardet.detect(file.read())["encoding"]

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

    metadata = metadata.rename(str.strip, axis = "columns")
    metadata = metadata.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    metadata.to_csv(newMetadataPath, "\t", index = False)

    return newMetadataPath
