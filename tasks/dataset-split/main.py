from pathlib import Path

import sys
import csv
import zipfile
import logging

from coretex import currentTaskRun
from coretex import NetworkDataset, ImageDataset, CustomDataset, SequenceDataset


def numberOfSamplesInEachNewDataset(n: int, numberOfNewDatasets: int) -> list[int]:
    r = n % numberOfNewDatasets
    numberOfSamples = [n // numberOfNewDatasets] * numberOfNewDatasets
    for i in range(r):
        numberOfSamples[i] += 1

    return numberOfSamples


def imageDatasetSplit(dataset: ImageDataset, numberOfSamples: list[int], projectID: int) -> list[NetworkDataset]:  
    dataset.download()
    samples = dataset.samples
    listOfNewDatasets: list[NetworkDataset] = []

    numberOfNewDatasets = len(numberOfSamples)
    
    counter = 0
    for i in range(numberOfNewDatasets):
        newDataset = ImageDataset.createDataset(f"{currentTaskRun().id}-newDataset-{i}", projectID)
        newDataset.saveClasses(dataset.classes)
        
        for _ in range(numberOfSamples[i]):
            samples[counter].unzip()
            newSample = newDataset.add(samples[counter].imagePath)
            
            tmpAnotation = samples[counter].load().annotation
            if(tmpAnotation is not None):
                newSample.saveAnnotation(tmpAnotation)
            
            try:
                tmpMetadata = samples[counter].loadMetadata()
                newSample.saveMetadata(tmpMetadata)
            except FileNotFoundError:
                logging.warning("File not found")
            except ValueError:
                logging.warning("Invalid data type")
            
            counter += 1

        listOfNewDatasets.append(newDataset)

    return listOfNewDatasets


def customDatasetSplit(dataset: CustomDataset, numberOfSamples: list[int], projectID: int) -> list[NetworkDataset]:  
    dataset.download()
    samples = dataset.samples
    listOfNewDatasets: list[NetworkDataset] = []

    numberOfNewDatasets = len(numberOfSamples)

    counter = 0
    for i in range(numberOfNewDatasets):
        newDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-newDataset-{i}", projectID)
        listOfNewDatasets.append(newDataset)

        for _ in range(numberOfSamples[i]):
            listOfNewDatasets[i].add(samples[counter].zipPath)
            counter += 1

    return listOfNewDatasets


def sequenceDatasetSplit(dataset: SequenceDataset, numberOfSamples: list[int], projectID: int) -> list[NetworkDataset]:
    dataset.download()
    md = list(dataset.metadata.load().folderContent)
    mdStrAdress = md[0]   #address where the file metadata is located in the form of a string
    
    samples = dataset.samples

    with open(mdStrAdress, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        listOfOriginalMetadata: list[dict] = []
        
        for row in reader:
            listOfOriginalMetadata.append(dict(row))

    listOfNewDatasets: list[NetworkDataset] = []

    numberOfNewDatasets = len(numberOfSamples)

    lastIndex = 0
    for i in range(numberOfNewDatasets):      
        listOfNewMetadata = listOfOriginalMetadata[lastIndex : lastIndex + numberOfSamples[i]]
        fieldNames = list(listOfNewMetadata[0].keys())
        lastIndex += numberOfSamples[i]

        with open(f"_metadata_{i}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldNames)
            writer.writeheader()
            writer.writerows(listOfNewMetadata)
            
        with zipfile.ZipFile(f"_metadata_{i}.zip", "w") as zipFile:
            zipFile.write(f"_metadata_{i}.csv")

        newDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-newDataset-{i}", projectID)
        if(newDataset is not None):
            listOfNewDatasets.append(newDataset)
        
        for j in range(numberOfSamples[i]):
            for sample in samples:
                sample.unzip()
                sampleNameInMetadata = listOfNewMetadata[j][fieldNames[0]]
                if(sample.name.startswith(sampleNameInMetadata.split("_")[0])):
                    listOfNewDatasets[i].add(sample.zipPath, sampleName=sampleNameInMetadata.split("_")[0])
        
        listOfNewDatasets[i].add(f"_metadata_{i}.zip")

        Path(f"_metadata_{i}.csv").unlink()
        Path(f"_metadata_{i}.zip").unlink()
        

    return listOfNewDatasets


def main() -> None:
    taskRun = currentTaskRun()
    dataset = taskRun.dataset
    
    numberOfNewDatasets = taskRun.parameters["numberOfNewDatasets"]
    projectID = taskRun.projectId

    n = dataset.count
    if(n <= numberOfNewDatasets or numberOfNewDatasets < 2):
        logging.error("Number of samples is smaller than the number you want to divide the database")
        sys.exit("The End")
        
    numberOfSamples = numberOfSamplesInEachNewDataset(n, numberOfNewDatasets)

    listOfNewDatasets: list[NetworkDataset]

    if(isinstance(dataset, ImageDataset)):
        listOfNewDatasets = imageDatasetSplit(dataset, numberOfSamples, projectID)
    
    if(isinstance(dataset, CustomDataset)):
        try:
            taskRun.setDatasetType(SequenceDataset)
            dataset = taskRun.dataset
            n = dataset.count
            numberOfSamples = numberOfSamplesInEachNewDataset(n, numberOfNewDatasets)
            listOfNewDatasets = sequenceDatasetSplit(dataset, numberOfSamples, projectID)
        except:
            listOfNewDatasets = customDatasetSplit(dataset, numberOfSamples, projectID) 
    

if __name__ == "__main__":
    main()
