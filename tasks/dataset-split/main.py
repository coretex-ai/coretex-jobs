import logging

from coretex import currentTaskRun, ImageDataset, CustomDataset, SequenceDataset, ImageSample, CustomSample, NetworkDataset
from coretex.networking import NetworkRequestError

from src.customDatasetSplit import customDatasetSplit
from src.imageDatasetSplit import imageDatasetSplit
from src.sequenceDatasetSplit import sequenceDatasetSplit


def main() -> None:
    taskRun = currentTaskRun()
    originalDataset = taskRun.dataset
    datasetParts = taskRun.parameters["datasetParts"]
    projectId = taskRun.projectId

    if originalDataset.count <= datasetParts:
        raise ValueError("Number of samples is smaller than the number you want to divide the dataset")
    if datasetParts < 2:
        raise ValueError("Dataset can be divided into at least two parts")

    splitDatasets: list[NetworkDataset]

    if isinstance(originalDataset, ImageDataset):
        logging.info(f">> [Dataset Split] Divisioning ImageDataset {originalDataset.name}...")
        splitDatasets = imageDatasetSplit(originalDataset, datasetParts, projectId)

    if isinstance(originalDataset, CustomDataset):
        try:
            taskRun.setDatasetType(SequenceDataset)
            originalDataset = taskRun.dataset
            logging.info(f">> [Dataset Split] Divisioning SequenceDataset {originalDataset.name}...")
            splitDatasets = sequenceDatasetSplit(originalDataset, datasetParts, projectId)
        
        except FileNotFoundError as e:
            logging.info(f">> [Dataset Split] Divisioning CustomDataset {originalDataset.name}...")
            splitDatasets = customDatasetSplit(originalDataset, datasetParts, projectId) 

    splitDatasetIDs = [ds.id for ds in splitDatasets]

    try:
        taskRun.submitOutput("splitDatasetIDs", splitDatasetIDs)
    except NetworkRequestError as e:
        logging.warning(f">> [Dataset Split] Error while submitting the output value: {e}")
        

if __name__ == "__main__":
    main()
