import logging
import sys


from coretex import currentTaskRun
from coretex import NetworkDataset, CustomDataset, ImageDataset

def allDatasetsIsSameClass(listOfDatasets: list[NetworkDataset]) -> bool:
    print([type(x) for x in listOfDatasets])
    print([isinstance(x, ImageDataset) for x in listOfDatasets])
    return len(set([isinstance(x, ImageDataset) for x in listOfDatasets])) == 1


#def allDatasetsIsSameType(listOfDatasets: list[NetworkDataset]) -> bool:
#    return len(set([type(x) for x in listOfDatasets])) == 1

"""
def customDatasetMerge(listOfDatasets: list[CustomDataset], projectID: int) -> NetworkDataset:
    newDataset: NetworkDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-merge-dataset", projectID)
    
    for dataset in listOfDatasets:
        dataset.download()
        samples = dataset.samples
        
        for sample in samples:
            newDataset.add(sample.zipPath)

    return newDataset
"""
def imageDatasetMerge(listOfDatasets: list[ImageDataset], projectID: int) -> NetworkDataset:
    mergeDataset = ImageDataset.createDataset(f"{currentTaskRun().id}-merge-dataset", projectID)
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
    listOfDatasets = taskRun.parameters["datasetsList"]
    projectID = taskRun.projectId
    
    mergeDataset: NetworkDataset

    
    

    if(not allDatasetsIsSameClass(listOfDatasets)):
        logging.error("The datasets are not of the same type.")
        sys.exit("The End")
    else:
        mergeDataset = imageDatasetMerge(listOfDatasets, projectID)
    
    #if(not allDatasetsIsSameType(listOfDatasets)):
    #    logging.error("The datasets are not of the same type.")
    #    sys.exit("The End")

  
    



if __name__ == "__main__":
    main()
