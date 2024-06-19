from typing import Any

import csv
import logging
import sys
import zipfile

from coretex import currentTaskRun, ImageDataset, CustomDataset, SequenceDataset, ImageSample, ImageDatasetClasses, CustomSample, SequenceSample

from utils import samplesSplit, SampleType, DatasetType


def imageDatasetSplit(splitSamples: list[list[ImageSample]], datasetClasses: ImageDatasetClasses, projectID: int) -> list[ImageDataset]:
    logging.info("Division of the ImageDataset")
    
    splitDatasetsList: list[ImageDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = ImageDataset.createDataset(f"{currentTaskRun().id}-newDataset-{index}", projectID)
        splitDataset.saveClasses(datasetClasses)

        for sample in sampleChunk:
            sample.unzip()
            addedSample = splitDataset.add(sample.imagePath)
            
            tmpAnotation = sample.load().annotation
            if tmpAnotation is not None:
                addedSample.saveAnnotation(tmpAnotation)

            try:
                tmpMetadata = sample.loadMetadata()
                addedSample.saveMetadata(tmpMetadata)
            except FileNotFoundError:
                logging.warning("Metadata file not found")
            except ValueError:
                logging.warning("Invalid metadata type")
        
        splitDatasetsList.append(splitDataset)

       

    return splitDatasetsList


def customDatasetSplit(splitSamples: list[list[CustomSample]], projectID: int) -> list[CustomDataset]:  
    splitDatasetsList: list[CustomDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-newDataset-{index}", projectID)

        for sample in sampleChunk:
            sample.unzip()
            splitDataset.add(sample.zipPath)

        splitDatasetsList.append(splitDataset)

        logging.info(f'New dataset named "{splitDataset.name}" has been created with {len(sampleChunk)} samples')

    return splitDatasetsList


def sequenceDatasetSplit(splitSamples: list[list[SequenceSample]], metadata: Any, projectID: int) -> list[CustomDataset]:
    logging.info("Division of the SequenceDataset")
    
    metadataAddress = list(metadata.load().folderContent)[0] #address where the file metadata is located in the form of a string
    
    with open(metadataAddress, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        originalMetadata: list[dict] = []
        
        for row in reader:
            originalMetadata.append(dict(row))

    splitDatasetsList: list[CustomDataset] = []
    
    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-newDataset-{index}", projectID)
        splitMetadataList: list[dict] = []
        
        for sample in sampleChunk:
            sample.unzip()
            splitDataset.add(sample.zipPath)
            
            fieldNames = list(originalMetadata[0].keys())
            
            for oneMetadata in originalMetadata:
                if sample.name.startswith(oneMetadata[fieldNames[0]].split("_")[0]):
                    splitMetadataList.append(oneMetadata)

        metadataCSV = f"_metadata_{index}.csv"
        with open(metadataCSV, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldNames)
            writer.writeheader()
            writer.writerows(splitMetadataList)

        metadataZIP = f"_metadata_{index}.zip"
        with zipfile.ZipFile(metadataZIP, "w") as zipFile:
            zipFile.write(metadataCSV)

        splitDataset.add(metadataZIP)

    splitDatasetsList.append(splitDataset)
          
    return splitDatasetsList


def main() -> None:
    taskRun = currentTaskRun()
    originalDataset = taskRun.dataset
    
    newDatasetCount = taskRun.parameters["numberOfNewDatasets"]
    projectID = taskRun.projectId

    n = originalDataset.count
    try:
        if n <= newDatasetCount:
            raise ValueError("Number of samples is smaller than the number you want to divide the dataset")
        if newDatasetCount < 2:
            raise ValueError("Dataset can be divided into at least two parts")

        splitSamples: list[list[SampleType]] = samplesSplit(originalDataset, newDatasetCount)

        splitDatasetsList: list[DatasetType]

        if isinstance(originalDataset, ImageDataset):
            splitDatasetsList = imageDatasetSplit(splitSamples, originalDataset.classes, projectID)

        if isinstance(originalDataset, CustomDataset):
            try:
                taskRun.setDatasetType(SequenceDataset)
                originalDataset = taskRun.dataset
                splitSamples = samplesSplit(originalDataset, newDatasetCount)
                listOfNewDatasets = sequenceDatasetSplit(splitSamples, originalDataset.metadata, projectID)
            except:
                logging.info(f"Divisioning CustomDataset {originalDataset.name}...")
                splitDatasetsList = customDatasetSplit(splitSamples, projectID) 
    
    except ValueError as e:
        logging.error(e)


if __name__ == "__main__":
    main()
