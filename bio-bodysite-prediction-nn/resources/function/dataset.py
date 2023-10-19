from typing import Generator
from pathlib import Path

import pickle

import tensorflow as tf
import numpy as np

from objects import Sample


def loadDataset(datasetPath: Path, uniqueTaxons: dict[str, int]) -> tf.data.Dataset:

    def generatorFunc() -> Generator:
        for path in datasetPath.iterdir():
            with path.open("rb") as file:
                sample: Sample = pickle.load(file)

            x = np.zeros((len(uniqueTaxons), ))

            for taxon in sample.taxons:
                x[uniqueTaxons[taxon.taxonId]] = np.log(taxon.count + 0.5)

            yield {
                "features": tf.convert_to_tensor(x, dtype = tf.float32)
            }

    return tf.data.Dataset.from_generator(
        generator = generatorFunc,
        output_signature = {
            "features": tf.TensorSpec(shape = (len(uniqueTaxons), ), dtype = tf.float32)
        }
    )
