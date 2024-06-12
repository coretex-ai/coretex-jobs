from pathlib import Path

import logging
import zipfile

import fitz
import ollama

from coretex import currentTaskRun, CustomDataset, TaskRun

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

        for pdfPath in sample.path.rglob("*.pdf"):
            if not "__MACOSX" in str(pdfPath):
                corpus.append(readPDF(pdfPath))

    return corpus


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()
    dataset = taskRun.dataset
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
        
        translatedText = ""
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
        
        txtFileName = f"{counter}.txt" 
        with open(txtFileName, "w") as file:
            file.write(translatedText)

        zipFileName = f"{counter}.zip"
        with zipfile.ZipFile(zipFileName, "w") as zipFile:
            zipFile.write(txtFileName)
        
        translatedDataset.add(zipFileName)


if __name__ == "__main__":
    main()
    