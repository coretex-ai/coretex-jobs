from typing import Generator
from pathlib import Path

import pickle

import tensorflow as tf
import numpy as np

from coretex import folder_manager

from .objects import Sample
from .utils import oneHotEncoding


def loadDataset(datasetPath: Path, uniqueBodySites: dict[str, int], uniqueTaxons: dict[str, int]) -> tf.data.Dataset:

    """
        Loads the dataset with TensorFlows data loading pipeline\n
        A logarithmic transformation will be aplied to the data

        Parameters
        ----------
        datasetPath : Path
            Path to the directory with all the samples of the dataset
        uniqueBodySites : dict[str, int]
            Dictionary mapping between bodysite names and their encodings
        uniqueTaxons : dict[str, int]
            Dictionary mapping between taxon ids and their encodings

        Returns
        -------
        tf.data.Dataset -> A TensorFlow dataset objcet representing the dataset
    """

    def generatorFunc() -> Generator:
        for path in datasetPath.iterdir():
            with path.open("rb") as file:
                sample: Sample = pickle.load(file)

            y = uniqueBodySites[sample.bodySite]
            x = np.zeros((len(uniqueTaxons), ))

            for taxon in sample.taxons:
                x[uniqueTaxons[taxon.taxonId]] = np.log(taxon.count + 0.5)

            yOneHot = oneHotEncoding(y, len(uniqueBodySites))
            yOneHot = yOneHot.reshape(len(uniqueBodySites), )

            yield {
                "features": tf.convert_to_tensor(x, dtype = tf.float32),
                "labels": tf.convert_to_tensor(yOneHot, dtype = tf.float32)
            }

    return tf.data.Dataset.from_generator(
        generator = generatorFunc,
        output_signature = {
            "features": tf.TensorSpec(shape = (len(uniqueTaxons), ), dtype = tf.float32),
            "labels": tf.TensorSpec(shape = (len(uniqueBodySites), ), dtype = tf.float32)
        }
    )


def createBatches(
    dataset: tf.data.Dataset,
    count: int,
    validationSplit: float,
    bufferSize: int,
    batchSize: int
) -> tuple[tf.data.Dataset, int, tf.data.Dataset, int]:

    trainCount = int((1 - validationSplit) * count)
    testCount = count - trainCount

    trainData = dataset.take(trainCount)
    testData = dataset.skip(trainCount).take(testCount)

    trainBatches = (
        trainData
        .cache(str(folder_manager.createTempFolder("tf_cache_file")))
        .shuffle(bufferSize)
        .batch(batchSize)
        .repeat()
        .prefetch(buffer_size = tf.data.AUTOTUNE)
    )
    testBatches = testData.batch(batchSize)

    return trainBatches, trainCount // batchSize, testBatches, count - trainCount
