from pathlib import Path

import tensorflow as tf

from coretex import Model


def loadSegmentationModel(ctxModel: Model) -> tf.lite.Interpreter:
    ctxModel.download()

    modelPath = ctxModel.path / "model.tflite"

    interpreter = tf.lite.Interpreter(str(modelPath))
    interpreter.allocate_tensors()

    return interpreter


def getWeights(model: Model) -> Path:
    model.download()

    return model.path / "model.pt"
