from typing import TypeVar

from coretex import NetworkSample


SampleType = TypeVar("SampleType", bound = NetworkSample)


def splitOriginalSamples(originalSamples: list[SampleType], datasetCount: int) -> list[list[SampleType]]:
    splitSamples: list[list[SampleType]] = [[] for _ in range(datasetCount)]

    for i in range(len(originalSamples)):
        splitSamples[i % datasetCount].append(originalSamples[i])

    return splitSamples
