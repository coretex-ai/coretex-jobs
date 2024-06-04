import logging

from coretex import currentTaskRun, NetworkDataset, CustomDataset, ImageDataset, SequenceDataset


def customDatasetMerge(listOfDatasets: list[CustomDataset], projectID: int) -> NetworkDataset:
    newDataset: NetworkDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-merge-custom-dataset", projectID)
    
    for dataset in listOfDatasets:
        dataset.download()
        samples = dataset.samples
        
        for sample in samples:
            newDataset.add(sample.zipPath)

    return newDataset


def imageDatasetMerge(listOfDatasets: list[ImageDataset], projectID: int) -> NetworkDataset:
    mergeDataset = ImageDataset.createDataset(f"{currentTaskRun().id}-merge-image-dataset", projectID)
    mergeDataset.saveClasses(listOfDatasets[0].classes)

    for dataset in listOfDatasets:
        dataset.download()
        samples = dataset.samples
        
        for sample in samples:
            sample.unzip()
            addedSample = mergeDataset.add(sample.imagePath)

            tmpAnotation = sample.load().annotation
            if(tmpAnotation is not None):
                addedSample.saveAnnotation(tmpAnotation)
            
            try:
                tmpMetadata = sample.loadMetadata()
                addedSample.saveMetadata(tmpMetadata)
            except FileNotFoundError:
                logging.warning("Metadata file not found")
            except ValueError:
                logging.warning("Invalid data type")

    return mergeDataset


def main() -> None:

    taskRun = currentTaskRun()
    projectID = taskRun.projectId
    listOfDatasets = taskRun.parameters["datasetsList"]

    if(isinstance(listOfDatasets[0], CustomDataset)):
        datasetMerge = customDatasetMerge(listOfDatasets, projectID)
    
    if(isinstance(listOfDatasets[0], ImageDataset)):
        datasetMerge = imageDatasetMerge(listOfDatasets, projectID)


if __name__ == "__main__":
    main()
