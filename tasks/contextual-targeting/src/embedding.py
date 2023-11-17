from typing import List

from sentence_transformers import SentenceTransformer

import numpy as np


model = SentenceTransformer("bert-base-nli-mean-tokens")


def generate(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy = True)  # type: ignore


def generateBatched(values: List[str], batchSize: int = 32) -> List[np.ndarray]:
    return model.encode(values, batchSize, convert_to_numpy = True)  # type: ignore


def compare(first: np.ndarray, second: np.ndarray) -> float:
    dotProduct: float = np.dot(first, second)
    firstMagnitude: float = np.sqrt(np.dot(first, first))
    secondMagnitude: float = np.sqrt(np.dot(second, second))

    return dotProduct / (firstMagnitude * secondMagnitude)
