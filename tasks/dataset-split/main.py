import logging

from coretex import currentTaskRun, ImageDataset, CustomDataset, SequenceDataset

from src.splitCustomDataset import splitCustomDataset
from src.splitImageDataset import splitImageDataset
from src.splitSequenceDataset import splitSequenceDataset
from src.utils import DatasetType


def main() -> None:
    taskRun = currentTaskRun()
    originalDataset = taskRun.dataset
    datasetParts = taskRun.parameters["datasetParts"]
    projectId = taskRun.projectId
    taskRunId = taskRun.id

    if originalDataset.count <= datasetParts:
        raise ValueError("Number of samples is smaller than the number you want to divide the dataset")

    if datasetParts < 2:
        raise ValueError("Dataset can be divided into at least two parts")

    splitDatasets: list[DatasetType]

    if isinstance(originalDataset, ImageDataset):
        logging.info(f">> [Dataset Split] Splitting ImageDataset {originalDataset.name}...")
        splitDatasets = splitImageDataset(originalDataset, datasetParts, taskRunId, projectId)

    if isinstance(originalDataset, CustomDataset):
        try:
            # If setDatasetType(SequenceDataset) cannot be executed, it raises a FileNotFoundError,
            # and then the dataset is split as CustomDataset.
            # The difference between SequenceDataset and CustomDataset is that
            # SequenceDataset requires a metadata file.

            taskRun.setDatasetType(SequenceDataset)
            originalDataset = taskRun.dataset
            logging.info(f">> [Dataset Split] Splitting SequenceDataset {originalDataset.name}...")
            splitDatasets = splitSequenceDataset(originalDataset, datasetParts, taskRunId, projectId)
        except FileNotFoundError:
            logging.info(f">> [Dataset Split] Splitting CustomDataset {originalDataset.name}...")
            splitDatasets = splitCustomDataset(originalDataset, datasetParts, taskRunId, projectId)

    outputDatasets = [dataset.id for dataset in splitDatasets]
    taskRun.submitOutput("outputDatasets", outputDatasets)


if __name__ == "__main__":
    main()
