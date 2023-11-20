from pathlib import Path

from ultralytics import YOLO

import tensorflow as tf

from coretex import Model


def loadSegmentationModel(ctxModel: Model) -> tf.lite.Interpreter:
    ctxModel.download()

    modelPath = ctxModel.path / "model.tflite"

    interpreter = tf.lite.Interpreter(str(modelPath))
    interpreter.allocate_tensors()

    return interpreter


def getObjectDetectionModel(model: Model) -> Path:
    model.download()

    return YOLO(model.path / "best.pt")
