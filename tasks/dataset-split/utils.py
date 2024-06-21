from typing import TypeVar

from coretex import NetworkDataset, NetworkSample


SampleType = TypeVar("SampleType", bound = NetworkSample)
DatasetType = TypeVar("DatasetType", bound = NetworkDataset)


def splitOriginalSamples(originalDataset: DatasetType, datasetCount: int) -> list[list[SampleType]]:
    originalDataset.download()
    originalSamples = originalDataset.samples

    splitSamples: list[list[SampleType]] = [[] for _ in range(datasetCount)]

    for i in range(originalDataset.count):
        splitSamples[i % datasetCount].append(originalSamples[i])
    
    return splitSamples
