from pathlib import Path

from coretex.utils import resizeWithPadding

import cv2
import numpy as np
import tensorflow as tf


def run(model: tf.lite.Interpreter, imagePath: Path) -> list[np.ndarray]:
    image = cv2.imread(str(imagePath))

    inputDetails = model.get_input_details()
    outputDetails = model.get_output_details()

    inputShape = inputDetails[0]["shape"]

    originalSize = (image.shape[1], image.shape[0])

    resized, pad = resizeWithPadding(image, (inputShape[2], inputShape[1]))
    normalized = resized / 255

    model.set_tensor(inputDetails[0]["index"], np.reshape(normalized, (1,) + normalized.shape).astype(np.float32))

    model.invoke()

    prediction = model.get_tensor(outputDetails[0]["index"])[0]

    # Crop out padding
    prediction = prediction[0 + pad[0]:prediction.shape[1] - pad[0], 0 + pad[1]:prediction.shape[0] - pad[1]]

    prediction = cv2.resize(prediction, originalSize)
    prediction = np.argmax(prediction, axis = -1)

    return  prediction
