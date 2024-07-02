from typing import Any
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pickle
import logging

from coretex import currentTaskRun, createDataset, CustomDataset, folder_manager

import ollama
import fitz
import faiss
import numpy as np

from src.model import launchOllamaServer, pullModel


OLLAMA_SERVER_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"


def readPDF(filePath: Path) -> list[str]:
    paragraphs: list[str] = []
    currentParagraph: list[str] = []
    with fitz.open(filePath) as doc:
        for page in doc:
            lines = page.get_text().split("\n")
            for line in lines:
                if line.strip() and (line[0].isupper() or line.startswith(" ")):
                    if currentParagraph and line[0].isupper():
                        paragraphs.append(" ".join(currentParagraph).strip())
                        currentParagraph = [line]
                    else:
                        currentParagraph.append(line)
                else:
                    currentParagraph.append(line)

            if currentParagraph:
                paragraphs.append(" ".join(currentParagraph).strip())
                currentParagraph = []

    return paragraphs


def createIndex(embeddings: np.ndarray) -> Any:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def loadCorpus(dataset: CustomDataset) -> np.ndarray:
    corpus: list[str] = []
    for sample in dataset.samples:
        sample.unzip()

        pdfPaths = [path for path in sample.path.iterdir() if path.suffix == ".pdf"]
        txtPaths = [path for path in sample.path.iterdir() if path.suffix == ".txt"]

        for pdfPath in pdfPaths:
            corpus.extend(readPDF(pdfPath))

        for txtPath in txtPaths:
            with open(txtPath, "r") as f:
                corpus.append(f.read())

    return np.array(corpus)


def main() -> None:
    taskRun = currentTaskRun()
    taskRun.dataset.download()

    serverProcess = launchOllamaServer()

    pullModel(EMBEDDING_MODEL)

    logging.info(">> [OllamaRAG] Loading text corpus")
    corpus = loadCorpus(taskRun.dataset)

    logging.info(">> [OllamaRAG] Embedding corpus")
    corpusEmbeddings = [ollama.embeddings(model = EMBEDDING_MODEL, prompt = chunk)["embedding"] for chunk in corpus]
    corpusEmbeddings = [embedding for embedding in corpusEmbeddings if len(embedding) != 0]

    logging.info(">> [OllamaRAG] Creating index")
    index = createIndex(np.array(corpusEmbeddings))

    datasetName = f"{taskRun.id}-rag-embedding-index"
    logging.info(f">> [OllamaRAG] Uploading to dataset \"{datasetName}\"")

    with createDataset(CustomDataset, datasetName, taskRun.projectId, {}) as outputDataset:
        corpusPath = folder_manager.temp / "corpus.pkl"
        with corpusPath.open("wb") as file:
            pickle.dump(corpus, file)

        indexPath = folder_manager.temp / "embeddings.index"
        faiss.write_index(index, str(indexPath))

        zipPath = Path("corpus-index.zip")
        with ZipFile(zipPath, "w", ZIP_DEFLATED) as archive:
            archive.write(corpusPath, corpusPath.name)
            archive.write(indexPath, indexPath.name)

        outputDataset.add(zipPath, zipPath.name)

    if serverProcess is not None:
        serverProcess.terminate()

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
