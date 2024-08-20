import os
import logging

from PIL import ImageColor
from keras.models import Model as KerasModel

import tensorflow as tf
import numpy as np
import cv2

from coretex import currentTaskRun, folder_manager, ImageDatasetClasses, CoretexImageAnnotation


def hasDotAnnotation(annotation: CoretexImageAnnotation) -> bool:
    for instance in annotation.instances:
        if any([len(segmentation) < 6 for segmentation in instance.segmentations]):
            return True

    return False


def createMask(predictionMask: np.ndarray) -> tf.Tensor:
    mask = tf.argmax(predictionMask, axis=-1)
    mask = mask[..., tf.newaxis]
    return mask[0]


def saveDatasetPredictions(group: str, model: KerasModel, dataset: tf.data.Dataset, classes: ImageDatasetClasses) -> None:
    predictions = model.predict(dataset)
    for index, prediction in enumerate(predictions):
        mask: np.ndarray = createMask(np.array([prediction])).numpy()
        coloredMask = np.empty(shape = (mask.shape[0], mask.shape[1], 3))

        for h, row in enumerate(mask):
            for w, pixel in enumerate(row):
                classId = pixel[0]

                if classId == 0:
                    coloredMask[h][w] = (0, 0, 0)
                else:
                    coloredMask[h][w] = ImageColor.getcolor(classes[classId - 1].color, "RGB")

        imageFileName = f"prediction_{index + 1}.png"
        imagePath = folder_manager.temp / imageFileName

        cv2.imwrite(str(imagePath), coloredMask)

        artifact = currentTaskRun().createArtifact(
            imagePath,
            os.path.join(group, imageFileName)
        )

        if artifact is None:
            logging.info(f">> [ImageSegmentation] Failed to create artifact for prediction image: {imageFileName}")
