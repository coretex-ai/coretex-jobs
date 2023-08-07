from typing import Optional, Tuple
from pathlib import Path

import csv

from coretex import CustomSample, CustomDataset
from coretex.bioinformatics import ctx_qiime2


def summarizeSample(sample: CustomSample, outputDir: Path) -> Path:
    demuxPath = sample.path / "demux.qza"
    visualizationPath = outputDir / "demux.qzv"

    ctx_qiime2.demuxSummarize(str(demuxPath), str(visualizationPath))
    return visualizationPath


def determineTruncLen(sample: CustomSample, forward: bool) -> int:
    sample.unzip()

    summariesFileName = "forward-seven-number-summaries.tsv" if forward else "reverse-seven-number-summaries.tsv"
    summariesTsv = list(sample.path.rglob(summariesFileName))[0]

    truncLen: Optional[int] = None
    with summariesTsv.open("r") as file:
        summaries = list(csv.reader(file, delimiter = "\t"))

    # summaries will allways have the median quality at row 5 of the csv
    medianQualitiesStr = summaries[5]
    medianQualitiesStr.pop(0)  # The first value will be "50%" and not a quality score
    medianQualities = [float(x) for x in medianQualitiesStr]

    highestScore = 0.0
    for index, qualityScore in enumerate(medianQualities):
        if qualityScore > highestScore:
            highestScore = qualityScore
            continue

        if qualityScore < highestScore * 0.7:
            truncLen = index
            break

    if not truncLen:
        raise RuntimeError(">> [Microbiome Analysis] Forward read truncLen could not be determined automatically")

    return truncLen


def loadPairedEnd(sample: CustomSample) -> Tuple[Path, Path, str]:
    sample.unzip()

    forwardPathList = list(sample.path.glob("*_R1_*.fastq"))
    reversePathList = list(sample.path.glob("*_R2_*.fastq"))

    if len(forwardPathList) != 1 and len(reversePathList) != 1:
        raise ValueError(f">> [Microbiome analysis] Invalid paired-end sample: {sample.name}. Must contain 2 files, one with \"_R1_\" and another with \"_R2\" in name")

    forwardPath = forwardPathList[0]
    reversePath = reversePathList[0]

    return forwardPath, reversePath, forwardPath.name.split("_")[0]


def isPairedEnd(dataset: CustomDataset) -> int:
    for sample in dataset.samples:
        sample.unzip()

        if sample.name.startswith("_metadata"):
            continue

        return sum([path.suffix == ".fastq" for path in sample.path.iterdir()]) == 2
