import logging

from coretex import CustomDataset, CustomSample

from .utils import splitOriginalSamples


def customDatasetSplit(originalDataset: CustomDataset, datasetParts: int, taskRunId: int, projectId: int) -> list[CustomDataset]:
    splitSamples: list[list[CustomSample]] = splitOriginalSamples(originalDataset.samples, datasetParts)

    splitDatasets: list[CustomDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = CustomDataset.createDataset(f"{taskRunId}-split-dataset-{index}", projectId)

        for sample in sampleChunk:
            splitDataset.add(sample.zipPath)
            logging.info(f">> [Dataset Split] The sample \"{sample.name}\" has been added to the dataset \"{splitDataset.name}\"")

        splitDatasets.append(splitDataset)

        logging.info(f">> [Dataset Split] New dataset named \"{splitDataset.name}\" contains {len(sampleChunk)} samples")

    return splitDatasets
