from pathlib import Path

import logging
import zipfile

import fitz
import ollama

from model import launchOllamaServer, pullModel, LLM

from coretex import currentTaskRun, CustomDataset


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
    
    translatedDataset = CustomDataset.createDataset(f"{taskRun.id}-translated", taskRun.projectId)

    language = taskRun.parameters["language"]

    for counter, document in enumerate(corpus, start = 1):
        document = [x.strip() for x in document]
        document = [line for line in document if line != ""]
        
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
            translatedText += answer + "\n"
        
        with open(f"{counter}.txt", "w") as file:
            file.write(translatedText)

        with zipfile.ZipFile(f"{counter}.zip", "w") as zipFile:
            zipFile.write(f"{counter}.txt")

        Path(f"{counter}.txt").unlink()
        filePath = f"{counter}.zip"
        
        translatedDataset.add(filePath)
        Path(f"{counter}.zip").unlink()


if __name__ == "__main__":
    main()
