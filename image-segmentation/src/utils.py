import os
import logging

from keras.models import Model as KerasModel

import tensorflow as tf
import numpy as np
import cv2

from coretex import ExecutingExperiment
from coretex.folder_management import FolderManager


def createMask(predictionMask: np.ndarray) -> tf.Tensor:
    mask = tf.argmax(predictionMask, axis=-1)
    mask = mask[..., tf.newaxis]
    return mask[0]


def saveDatasetPredictions(group: str, model: KerasModel, dataset: tf.data.Dataset) -> None:
    predictions = model.predict(dataset)
    for index, prediction in enumerate(predictions):
        mask = createMask([prediction])

        imageFileName = f"prediction_{index + 1}.png"
        imagePath = os.path.join(
            FolderManager.instance().temp,
            imageFileName
        )

        cv2.imwrite(imagePath, mask.numpy())

        artifact = ExecutingExperiment.current().createArtifact(
            imagePath,
            os.path.join(group, imageFileName)
        )

        if artifact is None:
            logging.info(f">> [ImageSegmentation] Failed to create artifact for prediction image: {imageFileName}")
