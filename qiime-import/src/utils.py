from typing import Optional
from pathlib import Path

import csv
import pandas as pd

from coretex import folder_manager

import chardet


CASEINSENSITIVE_NAMES = ["id", "sampleid", "sample id", "sample-id", "featureid" ,"feature id", "feature-id", "sample_id", "sample.id"]
CASESENSITIVE_NAMES = ["#SampleID" , "#Sample ID", "#OTUID", "#OTU ID", "sample_name"]


def columnNamePresent(metadataPath: Path, columnName: str) -> bool:
    with metadataPath.open("r") as metadata:
        for row in csv.reader(metadata, delimiter = "\t"):
            return columnName in row

    raise RuntimeError(">> [Microbiome Analysis] Metadata file is empty")


def detectFileEncoding(path: Path) -> Optional[str]:
    with path.open("rb") as file:
        return chardet.detect(file.read(10))["encoding"]


def convertMetadata(metadataPath: Path) -> Path:
    newMetadataPath = folder_manager.temp / f"{metadataPath.stem}.tsv"
    if metadataPath.suffix != ".csv" and metadataPath.suffix != ".tsv":
        raise ValueError(">> [Microbiome Analysis] Metadata has to be either tsv or csv")

    if metadataPath.suffix == ".csv":
        metadata = pd.read_csv(metadataPath, encoding = detectFileEncoding(metadataPath))
    else:
        metadata = pd.read_csv(metadataPath, encoding = detectFileEncoding(metadataPath), delimiter = "\t")

    for i, columnName in enumerate(metadata.columns):
        if columnName.lower() in CASEINSENSITIVE_NAMES or columnName in CASESENSITIVE_NAMES:
            break

        raise ValueError(f">> [Microbiome Analysis] Sample ID column not found. Recognized column names are: (case insensitive) - {CASEINSENSITIVE_NAMES}, (case sensitive) - {CASESENSITIVE_NAMES}")

    metadata.columns.values[i] = "sampleid"
    for sampleId in metadata["sampleid"]:
        sampleIdSplit = str(sampleId).split("_")
        if len(sampleIdSplit) > 1:
            metadata["sampleid"].replace(sampleId, sampleIdSplit[0], inplace = True)

    metadata.to_csv(newMetadataPath, "\t", index = False)

    return newMetadataPath
