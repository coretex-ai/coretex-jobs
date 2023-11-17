from pathlib import Path

import tensorflow as tf


def loadSegmentationModel(modelDir: Path) -> tf.lite.Interpreter:
    modelPath = modelDir / "model.tflite"

    interpreter = tf.lite.Interpreter(str(modelPath))
    interpreter.allocate_tensors()

    return interpreter


def getWeights(modelDir: Path) -> Path:
    return modelDir / "model.pt"
