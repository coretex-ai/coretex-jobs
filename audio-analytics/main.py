from typing import Optional

import logging

from coretex import CustomDataset, Run
from coretex.nlp import AudioTranscriber
from coretex.job import initializeJob

from src import text_search
from src.utils import createTranscriptionArtfacts, fetchModelFile
from src.occurence import NamedEntityRecognitionResult


def main(run: Run[CustomDataset]) -> None:
    pbmmUrl = run.parameters["modelPbmmUrl"]
    scorerUrl = run.parameters["modelScorerUrl"]
    pbmmName = "deepspeech-0.8.2-model.pbmm"
    scorerName = "deepspeech-0.8.2-model.scorer"

    logging.info(">> Downloading dataset and models from coretex web...")
    run.dataset.download()

    modelFile = fetchModelFile(pbmmUrl, pbmmName, ".pbmm")
    modelScorerFile = fetchModelFile(scorerUrl, scorerName, ".scorer")

    transcriber = AudioTranscriber(modelFile, modelScorerFile, parameters = {
        "batchSize": run.parameters["batchSize"],
        "modelName": pbmmName,
        "modelScorerName": scorerName
    })
    result = transcriber.transcribe(run.dataset, run.parameters["batchSize"])

    for sample, transcription in result:
        logging.info(f">> There are {len(transcription.tokens)} words in {sample.name}")

        coretexAudioResult: Optional[NamedEntityRecognitionResult] = None

        keywords = run.parameters.get("targetWords")
        if keywords is not None:
            logging.info(f">> Searching for words {keywords}...")

            targetWords = text_search.searchTranscription(transcription.tokens, keywords)
            coretexAudioResult = NamedEntityRecognitionResult.create(run.dataset.id, targetWords)

            for targetWord in targetWords:
                logging.info(f">> Found {len(targetWord.occurrences)} occurrences for \"{targetWord.text}\" word")

        logging.info(">> Creating artifacts...")
        createTranscriptionArtfacts(run, sample, transcription.text, transcription.tokens, coretexAudioResult)


if __name__ == "__main__":
    initializeJob(main)
