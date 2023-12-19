import logging

from coretex import ComputerVisionSample
from coretex.utils import resizeWithPadding

import cv2
import numpy as np
import tensorflow as tf


def run(model: tf.lite.Interpreter, sample: ComputerVisionSample) -> np.ndarray:
    logging.info(f">> [Document OCR] Running segmentation mask prediction on sample \"{sample.name}\"")
    sample.unzip()

    inputDetails = model.get_input_details()
    outputDetails = model.get_output_details()

    inputShape = inputDetails[0]["shape"]

    image = sample.load().image
    originalSize = (image.shape[1], image.shape[0])

    resized, verticalPadding, horizontalPadding = resizeWithPadding(image, inputShape[2], inputShape[1])
    normalized = resized / 255

    model.set_tensor(inputDetails[0]["index"], np.reshape(normalized, (1,) + normalized.shape).astype(np.float32))

    model.invoke()

    prediction = model.get_tensor(outputDetails[0]["index"])[0]

    # Crop out padding
    prediction = prediction[
        verticalPadding:prediction.shape[1] - verticalPadding,
        horizontalPadding:prediction.shape[0] - horizontalPadding
    ]

    prediction = cv2.resize(prediction, originalSize)
    prediction = np.argmax(prediction, axis = -1)

    return prediction
