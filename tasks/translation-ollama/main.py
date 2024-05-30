from coretex import currentTaskRun
from coretex import CustomDataset
from typing import Any

from pathlib import Path

import logging
import fitz
import ollama
import zipfile

import numpy as np

from model import launchOllamaServer, pullModel, LLM


def readPDF(filePath: Path) -> list[str]:
    pagesText: list[str] = []
    with fitz.open(filePath) as doc:
        for page in doc:
            paragraphs = page.get_text().split("\n")
            pagesText.extend(paragraphs)

    return pagesText


def loadCorpus(dataset: CustomDataset) -> list[list[str]]:
    corpus: list[list[str]] = []
    for sample in dataset.samples:
        sample.unzip()
        for pdfPath in sample.path.glob("*.pdf"):
            corpus.append(readPDF(pdfPath))

    return corpus


def main() -> None:
    taskRun = currentTaskRun()
    dataset: CustomDataset = taskRun.dataset
    dataset.download()

    launchOllamaServer()
    
    logging.info(">> [OllamaRAG] Pulling model")
    pullModel(LLM)
    
    logging.info(">> [OllamaRAG] Loading text corpus")
    corpus = loadCorpus(taskRun.dataset)
    
    newDatasetName = f"{taskRun.id}-translated"
    newDataset = CustomDataset.createDataset(newDatasetName, taskRun.projectId)

    language = taskRun.parameters["language"]

    for counter, document in enumerate(corpus, start=1):
        document = [x.strip() for x in document]
        while "" in document:
            document.remove("")

        translatedText: str = ""
        for paragraph in document:
            logging.info(">> [OllamaRAG] Translating paragraph")
            query = f"I will send you one paragraph, you translate into {language}. Let your response be only the translation of the sent paragraph, without additional comments. The paragraph to be translated is: {paragraph}"
            msg = {
                "role": "user",
                "content": query
            }
            response = ollama.chat(model = LLM, messages = [msg])
            answer = response["message"]["content"]
            translatedText = translatedText + answer + "\n"
        
        with open(f"{counter}.txt", "w") as file:
            file.write(translatedText)

        with zipfile.ZipFile(f"{counter}.zip", "w") as zipFile:
            zipFile.write(f"{counter}.txt")

        filePath = f"{counter}.zip"
        newDataset.add(filePath)


main()

