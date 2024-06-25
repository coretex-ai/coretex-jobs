from pathlib import Path

import csv
import logging
import zipfile

from coretex import currentTaskRun, folder_manager, SequenceDataset, NetworkDataset, CustomDataset, SequenceSample

import chardet

from .utils import splitOriginalSamples


CASEINSENSITIVE_NAMES = ["id", "sampleid", "sample id", "sample-id", "featureid" ,"feature id", "feature-id", "sample_id", "sample.id"]
CASESENSITIVE_NAMES = ["SampleID" , "Sample ID", "OTUID", "OTU ID", "sample_name"]


class MetadataFileNotFoundError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def detectEncodingCsv(filePath: Path) -> str:
    with open(filePath, "rb") as file:
        data = file.read()
        encoding = dict(chardet.detect(data))["encoding"]
        return str(encoding)
    

def checkSamplesInMetadata(originalSamples: list[SequenceSample], originalMetadata: list[dict[str, str]], fieldNames: list[str], indexId: int) -> list[SequenceSample]:
    prefixMetadata = [oneMetadata[fieldNames[indexId]].split("_")[0] for oneMetadata in originalMetadata]

    return [sample for sample in originalSamples if sample.name[:sample.name.rfind("-")] in prefixMetadata]


def checkMetadataInSamples(originalSamples: list[SequenceSample], originalMetadata: list[dict[str, str]], fieldNames: list[str], indexId: int) -> list[dict[str, str]]:
    prefixSamples = [sample.name[:sample.name.rfind("-")] for sample in originalSamples]
    
    return [metadata for metadata in originalMetadata if metadata[fieldNames[indexId]].split("_")[0] in prefixSamples]


def sequenceDatasetSplit(originalDataset: SequenceDataset, datasetParts: int, projectId: int) -> list[NetworkDataset]:
    metadataAddress = list(originalDataset.metadata.load().folderContent)[0]  # address where the file metadata is located in the form of a string
    if metadataAddress is None:
        raise MetadataFileNotFoundError("The metadata file was not found")

    encoding = detectEncodingCsv(metadataAddress)

    with open(metadataAddress, mode = "r", newline = "", encoding = encoding) as file:
        reader = csv.DictReader(file)
        
        fieldNames = reader.fieldnames
        if fieldNames is None:
            raise ValueError("Invalid metadata file")
        fieldNames = list(fieldNames)
        
        originalMetadata: list[dict[str, str]] = []
        for row in reader:
            originalMetadata.append(row)
    
    if not (any(x.lower in fieldNames for x in CASEINSENSITIVE_NAMES) or any(x in fieldNames for x in CASESENSITIVE_NAMES)):
        raise ValueError("Invalid metadata file. There is no valid column with sample names")
    
    for name in fieldNames:
        if name.lower() in CASEINSENSITIVE_NAMES or name in CASEINSENSITIVE_NAMES:
            indexId = fieldNames.index(name)
            break
    
    samples = checkSamplesInMetadata(originalDataset.samples, originalMetadata, fieldNames, indexId)
    if len(samples) <= datasetParts:
        raise ValueError("Number of valid samples is smaller than the number you want to divide the dataset")
    
    metadata = checkMetadataInSamples(originalDataset.samples, originalMetadata, fieldNames, indexId)
    splitSamples = splitOriginalSamples(samples, datasetParts)

    splitDatasets: list[NetworkDataset] = []
    
    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-split-dataset-{index}", projectId)
        
        splitMetadatas: list[dict[str, str]] = []
        for sample in sampleChunk:
            sample.unzip()
            splitDataset.add(sample.zipPath)
            logging.info(f">> [Dataset Split] The sample \"{sample.name}\" has been added to the dataset \"{splitDataset.name}\"")

            for oneMetadata in metadata:
                prefixSampleName = oneMetadata[fieldNames[indexId]].split("_")[0]
                if sample.name.startswith(prefixSampleName):
                    splitMetadatas.append(oneMetadata)
                
        csvMetadataName = f"metadata-{index}.csv"
        csvMetadata = folder_manager.temp / csvMetadataName
        with open(csvMetadata, "w", newline = "") as file:
            writer = csv.DictWriter(file, fieldnames=fieldNames)
            writer.writeheader()
            writer.writerows(splitMetadatas)

        zipMetadataName = f"metadata-{index}.zip"
        zipMetadata = folder_manager.temp / zipMetadataName
        with zipfile.ZipFile(zipMetadata, "w") as zipFile:
            zipFile.write(csvMetadata, zipMetadataName)

        splitDataset.add(zipMetadata)
        logging.info(f">> [Dataset Split] The _metadata sample \"{zipMetadataName}\" has been added to the dataset \"{splitDataset.name}\"")

        splitDatasets.append(splitDataset)
        logging.info(f">> [Dataset Split] New dataset named \"{splitDataset.name}\" contains {len(sampleChunk)} samples")
    
    return splitDatasets
