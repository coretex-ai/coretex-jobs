from typing import Optional
from zipfile import ZipFile
from pathlib import Path

import os
import json
import logging

from coretex import CustomSample, cache, TaskRun, folder_manager
from coretex.nlp import Token  # type: ignore[attr-defined]

from .occurence import NamedEntityRecognitionResult


def createTranscriptionArtfacts(
    taskRun: TaskRun,
    sample: CustomSample,
    transcribedText: str,
    tokens: list[Token],
    coretexAudioResult: Optional[NamedEntityRecognitionResult] = None
) -> None:

    tempFolder = folder_manager.createTempFolder(f"{sample.id}")

    textPath = os.path.join(tempFolder, "transcription.txt")
    transcriptionPath = os.path.join(tempFolder, "transcription.json")
    searchTranscriptionPath = os.path.join(tempFolder, "transcription.ner")

    # Transcription txt
    with open(textPath, "w") as transcriptionFile:
        transcriptionFile.write(transcribedText)

    if not taskRun.createArtifact(textPath, f"{sample.id}/transcription.txt"):
        logging.error(f">> Artifact upload failed for file {textPath}!")

    # Transcription words/json
    with open(transcriptionPath, "w") as transcriptionFile:
        json.dump([word.encode() for word in tokens], transcriptionFile, indent = 4)

    if not taskRun.createArtifact(transcriptionPath, f"{sample.id}/transcription.json"):
        logging.error(f">> Artifacts upload failed for file {transcriptionPath}!")

    # Text search result
    if coretexAudioResult is not None:
        with open(searchTranscriptionPath, "w") as searchResultFile:
            json.dump(coretexAudioResult.encode(), searchResultFile, indent = 4)

        if not taskRun.createArtifact(searchTranscriptionPath, f"{sample.id}/transcription.ner"):
            logging.error(f">> Artifacts upload failed for file {searchTranscriptionPath}!")


def fetchModelFile(modelUrl: str, modelName: str, modelSuffix: str) -> Path:
    if not cache.exists(modelUrl):
        cache.storeUrl(modelUrl, modelName)

    modelPath = cache.getPath(modelUrl)

    with ZipFile(modelPath, "r") as zipFile:
        zipFile.extractall(folder_manager.cache)

    return modelPath.with_suffix(modelSuffix)
