from typing import Any
from pathlib import Path

import pickle
import faiss
import numpy as np


def loadCorpusAndIndex(dirPath: Path) -> tuple[np.ndarray, Any]:
    with open(dirPath / "corpus.pkl", "rb") as file:
        corpus = pickle.load(file)

    index = faiss.read_index(str(dirPath / "embeddings.index"))

    return corpus, index


def retrieveDocuments(queryEmbedding: np.ndarray, index: faiss.IndexFlatL2, corpus: np.ndarray, k: int) -> list[tuple[str, int]]:
    distances, indices = index.search(queryEmbedding, k)
    return [(corpus[i], distances[0][j]) for j, i in enumerate(indices[0])]
