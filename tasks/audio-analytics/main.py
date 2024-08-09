from typing import Optional

import logging

from coretex import CustomDataset, TaskRun, currentTaskRun
from coretex.nlp import AudioTranscriber  # type: ignore[attr-defined]

from src import text_search
from src.utils import createTranscriptionArtfacts, fetchModelFile
from src.occurence import NamedEntityRecognitionResult


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    pbmmUrl = taskRun.parameters["modelPbmmUrl"]
    scorerUrl = taskRun.parameters["modelScorerUrl"]
    pbmmName = "deepspeech-0.8.2-model.pbmm"
    scorerName = "deepspeech-0.8.2-model.scorer"

    logging.info(">> Downloading dataset and models from coretex web...")
    taskRun.dataset.download()

    modelFile = fetchModelFile(pbmmUrl, pbmmName, ".pbmm")
    modelScorerFile = fetchModelFile(scorerUrl, scorerName, ".scorer")

    transcriber = AudioTranscriber(modelFile, modelScorerFile, parameters = {
        "batchSize": taskRun.parameters["batchSize"],
        "modelName": pbmmName,
        "modelScorerName": scorerName
    })
    result = transcriber.transcribe(taskRun.dataset, taskRun.parameters["batchSize"])

    for sample, transcription in result:
        logging.info(f">> There are {len(transcription.tokens)} words in {sample.name}")

        coretexAudioResult: Optional[NamedEntityRecognitionResult] = None

        keywords = taskRun.parameters.get("targetWords")
        if keywords is not None:
            logging.info(f">> Searching for words {keywords}...")

            targetWords = text_search.searchTranscription(transcription.tokens, keywords)
            coretexAudioResult = NamedEntityRecognitionResult.create(taskRun.dataset.id, targetWords)

            for targetWord in targetWords:
                logging.info(f">> Found {len(targetWord.occurrences)} occurrences for \"{targetWord.text}\" word")

        logging.info(">> Creating artifacts...")
        createTranscriptionArtfacts(taskRun, sample, transcription.text, transcription.tokens, coretexAudioResult)


if __name__ == "__main__":
    main()
