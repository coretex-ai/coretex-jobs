import os
import logging

from keras.models import Model as KerasModel

import tensorflow as tf
import numpy as np
import cv2

from coretex import currentTaskRun, folder_manager


def createMask(predictionMask: np.ndarray) -> tf.Tensor:
    mask = tf.argmax(predictionMask, axis=-1)
    mask = mask[..., tf.newaxis]
    return mask[0]


def saveDatasetPredictions(group: str, model: KerasModel, dataset: tf.data.Dataset) -> None:
    predictions = model.predict(dataset)
    for index, prediction in enumerate(predictions):
        mask = createMask([prediction])

        imageFileName = f"prediction_{index + 1}.png"
        imagePath = folder_manager.temp / imageFileName

        cv2.imwrite(str(imagePath), mask.numpy())

        artifact = currentTaskRun().createArtifact(
            imagePath,
            os.path.join(group, imageFileName)
        )

        if artifact is None:
            logging.info(f">> [ImageSegmentation] Failed to create artifact for prediction image: {imageFileName}")
