from typing import Optional, Any

import logging

from keras.callbacks import Callback

from coretex import currentTaskRun


class DisplayCallback(Callback):  # type: ignore[misc]

    def __init__(self, epochs: int) -> None:
        super().__init__()

        self.epochs = epochs

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        if logs is None:
            return

        loss = logs["loss"]
        valLoss = logs["val_loss"]

        accuracy = logs["accuracy"]
        valAccuracy = logs["val_accuracy"]

        logging.info(f">> [Image Segmentation] Finished epoch {epoch + 1}/{self.epochs}")
        logging.info(f"\tLoss - train: {loss}, val: {valLoss}")
        logging.info(f"\tAccuracy - train: {accuracy}, val: {valAccuracy}")

        if not currentTaskRun().submitMetrics({
            "loss": (epoch + 1, loss),
            "accuracy": (epoch + 1, accuracy),
            "val_loss": (epoch + 1, valLoss),
            "val_accuracy": (epoch + 1, valAccuracy)
        }):
            logging.warning(">> [Image Segmentation] Failed to submit metrics!")
