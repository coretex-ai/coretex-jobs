import logging

from coretex import currentTaskRun, NetworkDataset, CustomDataset, CustomSample

from .utils import splitOriginalSamples


def customDatasetSplit(originalDataset: CustomDataset, datasetParts: int, projectId: int) -> list[NetworkDataset]:  
    splitSamples: list[list[CustomSample]] = splitOriginalSamples(originalDataset.samples, datasetParts)
    
    splitDatasets: list[NetworkDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-split-dataset-{index}", projectId)

        for sample in sampleChunk:
            splitDataset.add(sample.zipPath)
            logging.info(f">> [Dataset Split] The sample \"{sample.name}\" has been added to the dataset \"{splitDataset.name}\"")

        splitDatasets.append(splitDataset)

        logging.info(f">> [Dataset Split] New dataset named \"{splitDataset.name}\" contains {len(sampleChunk)} samples")

    return splitDatasets
