from typing import TypeVar

from coretex import NetworkDataset, NetworkSample

SampleType = TypeVar("SampleType", bound = NetworkSample)
DatasetType = TypeVar("DatasetType", bound = NetworkDataset)


def samplesSplit(originalDataset: DatasetType, datasetCount: int) -> list[list[SampleType]]:
    originalDataset.download()
    originalSamples = originalDataset.samples

    splittedSamples: list[list[SampleType]] = [[] for _ in range(datasetCount)]

    for i in range(originalDataset.count):
        splittedSamples[i % datasetCount].append(originalSamples[i])
    
    return splittedSamples
