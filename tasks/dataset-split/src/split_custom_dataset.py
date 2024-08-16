import logging

from coretex import CustomDataset, CustomSample

from .utils import splitOriginalSamples


def splitCustomDataset(originalDataset: CustomDataset, datasetParts: int, projectId: int) -> list[CustomDataset]:
    splitSamples: list[list[CustomSample]] = splitOriginalSamples(originalDataset.samples, datasetParts)

    splitDatasets: list[CustomDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        dependencies = [str(originalDataset.id), str(datasetParts), str(projectId), str(index)]
        try:
            splitDataset = CustomDataset.fetchCachedDataset(dependencies)
            logging.info(f">> [Dataset Split] The dataset with the name \"{splitDataset.name}\" has been fetched ")
        except ValueError:
            splitDataset = CustomDataset.createCacheDataset("split-dataset", dependencies, projectId)
            for sample in sampleChunk:
                splitDataset.add(sample.zipPath)
                logging.info(f">> [Dataset Split] The sample \"{sample.name}\" has been added to the dataset \"{splitDataset.name}\"")

        splitDatasets.append(splitDataset)

        logging.info(f">> [Dataset Split] New dataset named \"{splitDataset.name}\" contains {splitDataset.count} samples")

    return splitDatasets
