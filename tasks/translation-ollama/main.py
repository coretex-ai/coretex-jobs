from pathlib import Path

import logging
import zipfile

import fitz  # PyMyPDF
import ollama

from coretex import currentTaskRun, folder_manager, CustomDataset, TaskRun

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

        pdfPaths = list(sample.path.rglob("*.pdf"))
        if len(pdfPaths) == 0:
            raise ValueError(">> [LLM Translate] The provided dataset does not contain any .pdf documents")
        
        for pdfPath in pdfPaths:
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
            query = f"I will send you one paragraph, you translate into {language}. \
                Let your response be only the translation of the sent paragraph, \
                without additional comments. The paragraph to be translated is: {paragraph}"
            msg = {
                "role": "user",
                "content": query
            }
            response = ollama.chat(model = LLM, messages = [msg])
            answer = response["message"]["content"]
            translatedText += answer + "\n"
        
        txtFileName = f"file-{counter}.txt"
        txtFile = folder_manager.temp / txtFileName
        with open(txtFile, "w") as f:
            f.write(translatedText)

        zipFileName = f"file-{counter}.zip"
        zipFile = folder_manager.temp / zipFileName
        with zipfile.ZipFile(zipFile, "w") as zf:
            zf.write(txtFile, txtFileName)
        
        translatedDataset.add(zipFile)

        taskRun.submitOutput("translatedDataset", translatedDataset)

if __name__ == "__main__":
    main()
