from pathlib import Path

import logging

from Bio import SeqIO

import matplotlib.pyplot as plt

from coretex import currentTaskRun, SequenceDataset, folder_manager


def calculateAverageScores(qualityScores: list[list[int]]) -> list[float]:
    maxLength = max(len(readScores) for readScores in qualityScores)

    totalScores = [0] * maxLength
    for readScores in qualityScores:
        for i in range(len(readScores)):
            totalScores[i] += readScores[i]

    return [score / len(qualityScores) for score in totalScores]


def analyzeFastq(sequencePath: Path) -> list[float]:
    qualityScores: list[list[int]] = []
    with sequencePath.open("r") as file:
        for record in SeqIO.parse(file, "fastq"):
            qualityScores.append(record.letter_annotations["phred_quality"])

    return calculateAverageScores(qualityScores)


def createPlot(scores: list[float], forward: bool) -> Path:
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(range(len(scores)), scores, linestyle = "-", color = "b", linewidth = 2, label = "Phred Scores")

    if forward:
        title = "Average Forward Read Quality Scores"
        plotPath = folder_manager.temp / "forward_qualities.png"
    else:
        title = "Average Reverse Read Quality Scores"
        plotPath = folder_manager.temp / "reverse_qualities.png"

    ax.set_title(title, fontsize = 18)
    ax.set_xlabel("Base Pair Index", fontsize = 14)
    ax.set_ylabel("Phred Quality Score", fontsize = 14)
    ax.set_ylim(0, 40)
    ax.tick_params(axis = "both", which = "major", labelsize = 12)
    ax.grid(True, linestyle = "--", alpha = 0.5)

    ax.legend(loc = "upper right", fontsize = 12)

    plt.savefig(plotPath)

    return plotPath


def main() -> None:
    taskRun = currentTaskRun()
    taskRun.setDatasetType(SequenceDataset)
    dataset = taskRun.dataset
    dataset.download()

    forwardScores: list[list[float]] = []
    reverseScores: list[list[float]] = []

    for sample in dataset.samples:
        logging.info(f">> [Quality Scores] Analysing sample \"{sample.name}\"")
        sample.unzip()

        forwardScores.append(analyzeFastq(sample.forwardPath))
        reverseScores.append(analyzeFastq(sample.reversePath))

    forwardAverage = calculateAverageScores(forwardScores)
    reverseAverage = calculateAverageScores(reverseScores)

    forwardPlot = createPlot(forwardAverage, True)
    reversePlot = createPlot(reverseAverage, False)

    logging.info(">> [Quality Scores] Uploading plots")
    taskRun.createArtifact(forwardPlot, forwardPlot.name)
    taskRun.createArtifact(reversePlot, reversePlot.name)


if __name__ == "__main__":
    main()
