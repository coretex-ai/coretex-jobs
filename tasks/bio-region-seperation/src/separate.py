from pathlib import Path
from io import TextIOWrapper

import logging

from coretex.bioinformatics import sequence_alignment as sa

from .filepaths import SAMTOOLS


def argmax(array: list) -> int:
    return max(range(len(array)), key = lambda x : array[x])


def splitToFiles(inputFile: Path, readClasses: list[int], groups: list[Path]) -> None:
    outFiles: list[TextIOWrapper] = []
    for group in groups:
        outFiles.append(open(group / inputFile.name, "a"))

    readIndex = -1
    readClass = 0

    with inputFile.open("rb") as inFile:
        for lineIndex, line in enumerate(inFile):
            decodedLine = line.decode("utf-8")
            if lineIndex % 4 == 0:
                readIndex += 1
                readClass = readClasses[readIndex]

            outFiles[readClass].write(decodedLine)


def separate(bamDir: Path, inputFile: Path, groups: list[Path], thresholds: list[int]) -> None:
    scores: list[list[int]] = []
    positions: list[list[int]] = []

    logging.info(">> [Region Separation] Extracting data from BAM")
    for filePath in bamDir.iterdir():
        score, pos, lengths = sa.extractData(Path(SAMTOOLS), filePath)
        scores.append(score)
        positions.append(pos)

    finalPositions: list[int] = []

    logging.info(">> [Region Separation] Determining read positions")
    for readScores, readPositions in zip(list(zip(*scores)), list(zip(*positions))):
        finalPositions.append(readPositions[argmax(list(readScores))])

    readClasses: list[int] = []

    logging.info(">> [Region Separation] Calculating group identity")
    for position, length in zip(finalPositions, lengths):
        overlapIndices = range(position, position + length)
        regionScores: list[int] = []
        for region in range(len(thresholds) - 1):
            thresholds[-1] = length

            regionScore = 0
            for index in overlapIndices:
                if index > thresholds[region] and index < thresholds[region + 1]:
                    regionScore += 1

            regionScores.append(regionScore)

        readClasses.append(argmax(regionScores))

    logging.info(">> [Region Separation] Splitting file into separate folder")
    splitToFiles(inputFile, readClasses, groups)
