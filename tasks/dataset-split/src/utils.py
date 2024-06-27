from typing import TypeVar

import re

from coretex import NetworkSample


SampleType = TypeVar("SampleType", bound = NetworkSample)


def splitOriginalSamples(originalSamples: list[SampleType], datasetCount: int) -> list[list[SampleType]]:
    splitSamples: list[list[SampleType]] = [[] for _ in range(datasetCount)]

    for i in range(len(originalSamples)):
        splitSamples[i % datasetCount].append(originalSamples[i])

    return splitSamples

def validateAndCorectEntityName(name: str) -> str:
    pattern = r"^[a-z0-9-]{3,50}$"
    if re.fullmatch(pattern, name) is not None:
        return name

    name = name.lower()
    name = name.replace("_", "-")
    name = re.sub(r"[^a-z0-9-]", "", name)

    if len(name) < 3:
        name = name.ljust(3, "-")
    if len(name) > 50:
        name = name[:50]

    return name
