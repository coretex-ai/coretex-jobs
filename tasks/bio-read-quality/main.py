from pathlib import Path

import logging

from Bio import SeqIO

import matplotlib.pyplot as plt

from coretex import currentTaskRun, SequenceDataset, folder_manager, TaskRun


def calculateAverageScores(qualityScores: list[list[int]]) -> list[int]:
    maxLength = max(len(readScores) for readScores in qualityScores)

    totalScores = [0] * maxLength
    for readScores in qualityScores:
        for i in range(len(readScores)):
            totalScores[i] += readScores[i]

    return [int(score / len(qualityScores)) for score in totalScores]


def analyzeFastq(sequencePath: Path) -> list[int]:
    qualityScores: list[list[int]] = []
    with sequencePath.open("r") as file:
        for record in SeqIO.parse(file, "fastq"):
            qualityScores.append(record.letter_annotations["phred_quality"])

    return calculateAverageScores(qualityScores)


def createPlot(scores: list[int], title: str, plotPath: Path) -> Path:
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(range(len(scores)), scores, linestyle = "-", color = "b", linewidth = 2, label = "Phred Scores")

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
    taskRun: TaskRun[SequenceDataset] = currentTaskRun()
    taskRun.setDatasetType(SequenceDataset)
    taskRun.dataset.download()

    forwardScores: list[list[int]] = []
    reverseScores: list[list[int]] = []

    for sample in taskRun.dataset.samples:
        logging.info(f">> [Quality Scores] Analysing sample \"{sample.name}\"")
        sample.unzip()

        forwardScores.append(analyzeFastq(sample.forwardPath))
        reverseScores.append(analyzeFastq(sample.reversePath))

    logging.info(">> [Quality Scores] Calculating average scores")
    forwardAverage = calculateAverageScores(forwardScores)
    reverseAverage = calculateAverageScores(reverseScores)

    forwardPlot = createPlot(
        forwardAverage,
        "Average Forward Read Quality Scores",
        folder_manager.temp / "forward_qualities.png"
    )

    reversePlot = createPlot(
        reverseAverage,
        "Average Reverse Read Quality Scores",
        folder_manager.temp / "reverse_qualities.png"
    )

    logging.info(">> [Quality Scores] Uploading plots")
    taskRun.createArtifact(forwardPlot, forwardPlot.name)
    taskRun.createArtifact(reversePlot, reversePlot.name)


if __name__ == "__main__":
    main()
