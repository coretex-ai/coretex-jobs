from typing import Optional, Tuple
from pathlib import Path

import csv

from coretex import CustomSample, CustomDataset, Experiment
from coretex.bioinformatics import ctx_qiime2


def summarizeSample(sample: CustomSample, outputDir: Path) -> Path:
    demuxPath = Path(sample.path) / "demux.qza"
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
        qualityScore = float(qualityScore)
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
    for filePath in sample.path.iterdir():
        if filePath.suffix != ".fastq":
            continue

        if filePath.name.find("_R1_") != -1:
            forwardPath = filePath
        elif filePath.name.find("_R2_") != -1:
            reversePath = filePath
        else:
            raise ValueError(f">> [Microbiome analysis] Not found _R1_ or _R2_, indicting forward and reverse reads, in the file {filePath}")

    return forwardPath, reversePath, forwardPath.name[0: forwardPath.name.find("_R1_")]


def isPairedEnd(dataset: CustomDataset, experiment: Experiment):
    for sample in dataset.samples:
        sample.unzip()
        if any([path.name == experiment.parameters["metadataFileName"] for path in sample.path.iterdir()]):
            continue

        return sum([path.suffix == ".fastq" for path in sample.path.iterdir()]) == 2
