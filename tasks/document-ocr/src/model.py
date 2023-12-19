from coretex import Model
from ultralytics import YOLO

import tensorflow as tf


def loadSegmentationModel(ctxModel: Model) -> tf.lite.Interpreter:
    ctxModel.download()

    modelPath = ctxModel.path / "model.tflite"

    interpreter = tf.lite.Interpreter(str(modelPath))
    interpreter.allocate_tensors()

    return interpreter


def loadDetectionModel(model: Model) -> YOLO:
    model.download()

    return YOLO(model.path / "best.pt")
