from pathlib import Path

import tensorflow as tf


def loadModel(modelPath: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(str(modelPath))
    interpreter.allocate_tensors()

    return interpreter
