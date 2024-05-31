import sys
import csv
import zipfile
import logging

from coretex import currentTaskRun
from coretex import NetworkDataset, ImageDataset, CustomDataset, SequenceDataset


def numberOfElementsInEachNewDataset(n: int, numSplit: int) -> list[int]:
    r = n % numSplit
    numbers = [n // numSplit] * numSplit
    for i in range(r):
        numbers[i] += 1

    return numbers


def imageDatasetSplit(dataset: ImageDataset, numbers: list[int], projectID: int) -> list[NetworkDataset]:  
    dataset.download()
    samples = dataset.samples
    newDatasetsList: list[NetworkDataset] = []

    numSplit = len(numbers)
    
    counter = 0
    for i in range(numSplit):
        newDataset = ImageDataset.createDataset(f"{currentTaskRun().id}-newDataset-{i}", projectID)
        newDataset.saveClasses(dataset.classes)
        
        for _ in range(numbers[i]):
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

        newDatasetsList.append(newDataset)

    return newDatasetsList


def customDatasetSplit(dataset: CustomDataset, numbers: list[int], projectID: int) -> list[NetworkDataset]:  
    dataset.download()
    samples = dataset.samples
    newDatasetsList: list[NetworkDataset] = []

    numSplit = len(numbers)

    counter = 0
    for i in range(numSplit):
        newDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-newDataset-{i}", projectID)
        newDatasetsList.append(newDataset)

        for _ in range(numbers[i]):
            newDatasetsList[i].add(samples[counter].zipPath)
            counter += 1

    return newDatasetsList


def sequenceDatasetSplit(dataset: SequenceDataset, numbers: list[int], projectID: int) -> list[NetworkDataset]:
    dataset.download()
    md = list(dataset.metadata.load().folderContent)
    mdStrAdress = md[0]
    
    samples = dataset.samples

    with open(mdStrAdress, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        metaDataList: list[dict] = []
        for row in reader:
            metaDataList.append(dict(row))

    newDatasetsList: list[NetworkDataset] = []

    numSplit = len(numbers)

    lastIndex = 0
    for i in range(numSplit):      
        newMetadataList = metaDataList[lastIndex : lastIndex + numbers[i]]
        fieldNames = list(newMetadataList[0].keys())

        lastIndex += numbers[i]
        with open(f"_metadata_{i}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldNames)
            writer.writeheader()
            writer.writerows(newMetadataList)
            
        with zipfile.ZipFile(f"_metadata_{i}.zip", "w") as zipFile:
            zipFile.write(f"_metadata_{i}.csv")
        
        pathToMetadata = f"_metadata_{i}.zip"
        newDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-newDataset-{i}", projectID)
        if(newDataset is not None):
            newDatasetsList.append(newDataset)
        
        for j in range(numbers[i]):
            for sample in samples:
                sample.unzip()
                sampleNameInMetadata = newMetadataList[j][fieldNames[0]]
                if(sample.name.startswith(sampleNameInMetadata.split("_")[0])):
                    newDatasetsList[i].add(sample.zipPath, sampleName=sampleNameInMetadata.split("_")[0])
        
        newDatasetsList[i].add(pathToMetadata)

    return newDatasetsList


def main() -> None:
    taskRun = currentTaskRun()
    dataset = taskRun.dataset
    
    numSplit = taskRun.parameters["numSplit"]
    projectID = taskRun.projectId

    n = dataset.count
    
    if(n <= numSplit or numSplit < 2):
        logging.error("Number of samples is smaller than the number you want to divide the database")
        sys.exit("The End")
        
    numbers = numberOfElementsInEachNewDataset(n, numSplit)

    newDatasetsList: list[NetworkDataset]

    if(isinstance(dataset, ImageDataset)):
        newDatasetsList = imageDatasetSplit(dataset, numbers, projectID)
    
    if(isinstance(dataset, CustomDataset)):
        try:
            taskRun.setDatasetType(SequenceDataset)
            dataset = taskRun.dataset
            n = dataset.count
            numbers = numberOfElementsInEachNewDataset(n, numSplit)
            newDatasetsList = sequenceDatasetSplit(dataset, numbers, projectID)
        except:
            newDatasetsList = customDatasetSplit(dataset, numbers, projectID) 
    

main()

