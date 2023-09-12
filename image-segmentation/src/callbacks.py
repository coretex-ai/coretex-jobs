from typing import Final

import time
import logging

from keras.models import Model as KerasModel
from keras.callbacks import Callback

import tensorflow as tf

from coretex import currentExperiment

from .utils import saveDatasetPredictions


def timeDiff(value: float, decimalPlaces: int = 4) -> float:
    return round(time.time() - value, decimalPlaces)


class DisplayCallback(Callback):

    def __init__(self, model: KerasModel, dataset: tf.data.Dataset, epochs: int) -> None:
        super().__init__()

        self.__model: Final = model
        self.__dataset: Final = dataset
        self.__epochs: Final = epochs

    def on_train_begin(self, logs = None):
        self.trainBegin = time.time()
        logging.info(">> [ImageSegmentation] Started training")

    def on_train_end(self, logs = None):
        logging.info(f">> [ImageSegmentation] Finished training in {timeDiff(self.trainBegin)}")

    def on_epoch_begin(self, epoch: int, logs = None):
        self.epochBegin = time.time()
        logging.info(f">> [ImageSegmentation] Started epoch {epoch + 1}/{self.__epochs}")

    def on_epoch_end(self, epoch: int, logs = None):
        if logs is not None:
            if not currentExperiment().submitMetrics({
                "loss": (epoch + 1, logs["loss"]),
                "accuracy": (epoch + 1, logs["accuracy"])
            }):
                logging.warning(">> [BMSTraining] Failed to submit metrics!")

        saveDatasetPredictions(f"After epoch {epoch + 1}", self.__model, self.__dataset)
        logging.info(f">> [ImageSegmentation] Finished epoch {epoch + 1}/{self.__epochs} in {timeDiff(self.epochBegin)}")

    def on_train_batch_begin(self, batch: int, logs = None):
        self.trainBatchBegin = time.time()
        logging.info(f">> [ImageSegmentation] Started train batch {batch + 1}")

    def on_train_batch_end(self, batch, logs = None):
        logging.info(f">> [ImageSegmentation] Finished train batch {batch + 1} in {timeDiff(self.trainBatchBegin)}")

    def on_test_begin(self, logs = None):
        self.testBegin = time.time()
        logging.info(">> [ImageSegmentation] Started test")

    def on_test_end(self, logs = None):
        logging.info(f">> [ImageSegmentation] Finished test in {timeDiff(self.testBegin)}")

    def on_test_batch_begin(self, batch: int, logs = None):
        self.testBatchBegin = time.time()
        logging.info(f">> [ImageSegmentation] Started test batch {batch + 1}")

    def on_test_batch_end(self, batch: int, logs = None):
        logging.info(f">> [ImageSegmentation] Finished test batch {batch + 1} in {timeDiff(self.testBatchBegin)}")

    def on_predict_begin(self, logs = None):
        self.predictBegin = time.time()
        logging.info(">> [ImageSegmentation] Started predict")

    def on_predict_end(self, logs = None):
        logging.info(f">> [ImageSegmentation] Finished predict in {timeDiff(self.predictBegin)}")

    def on_predict_batch_begin(self, batch: int, logs = None):
        self.predictBatchBegin = time.time()
        logging.info(f">> [ImageSegmentation] Started predict batch {batch + 1}")

    def on_predict_batch_end(self, batch: int, logs = None):
        logging.info(f">> [ImageSegmentation] Finished predict batch {batch + 1} in {timeDiff(self.predictBatchBegin)}")
