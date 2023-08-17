
from pathlib import Path

import csv

from coretex import folder_manager


CASEINSENSITIVE_NAMES = ["id", "sampleid", "sample id", "sample-id", "featureid" ,"feature id", "feature-id"]
CASESENSITIVE_NAMES = ["#SampleID" , "#Sample ID", "#OTUID", "#OTU ID", "sample_name"]


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
