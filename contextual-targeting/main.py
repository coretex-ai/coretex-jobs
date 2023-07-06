from typing import Dict, Any, Tuple, List
from pathlib import Path

import logging

from coretex import CustomDataset, ExecutingExperiment, CustomSample
from coretex.nlp import AudioTranscriber, Transcription
from coretex.project import initializeProject
from coretex.folder_management import FolderManager

import matplotlib.pyplot as plt

from src.utils import fetchModelFile

import src.embedding as embedding


MODEL_NAME = "deepspeech-0.8.2-model.pbmm"
MODEL_SCORER_NAME = "deepspeech-0.8.2-model.scorer"


def transcribe(dataset: CustomDataset, parameters: Dict[str, Any]) -> List[Tuple[CustomSample, Transcription]]:
    modelFile = fetchModelFile(parameters["modelUrl"], MODEL_NAME, ".pbmm")
    modelScorerFile = fetchModelFile(parameters["modelScorerUrl"], MODEL_SCORER_NAME, ".scorer")

    transcriber = AudioTranscriber(modelFile, modelScorerFile, parameters = {
        "batchSize": parameters["batchSize"],
        "modelName": MODEL_NAME,
        "modelScorerName": MODEL_SCORER_NAME
    })

    return transcriber.transcribe(dataset, parameters["batchSize"])


def plotResults(experiment: ExecutingExperiment, results: Dict[int, float], directory: Path) -> None:
    x = list(results.keys())
    y = [results[key] for key in x]

    chartPath = directory / "results.png"

    _, ax = plt.subplots()
    ax.locator_params("x", True, integer = True)
    ax.ticklabel_format(useOffset = False, style = "plain")
    ax.plot(x, y)

    plt.savefig(chartPath)
    plt.close()

    if not experiment.createArtifact(str(chartPath), "results.png"):
        logging.error(">> Failed to save artifact for: results.png")


def main(experiment: ExecutingExperiment[CustomDataset]) -> None:
    logging.info(">> Downloading dataset and models from coretex web...")
    experiment.dataset.download()

    logging.info(">> Generating embedding for target")
    targetEmbedding = embedding.generate(experiment.parameters["target"])

    results: Dict[int, float] = {}

    # will perform transcription only on audio samples, text samples are only tokenized
    for sample, transcription in transcribe(experiment.dataset, experiment.parameters):
        logging.info(f">> Generating embeddings for sample: {sample.name}")
        textEmbedding = embedding.generate(transcription.text)

        logging.info(f">> Comparing sample: {sample.name} to target")
        similarity = embedding.compare(textEmbedding, targetEmbedding)

        logging.info(f">> sample: {sample.name} is {(similarity * 100):.2f} % similar to target")
        results[sample.id] = similarity

    chartDirectory = Path(FolderManager.instance().createTempFolder("charts"))
    plotResults(experiment, results, chartDirectory)


if __name__ == "__main__":
    initializeProject(main)
