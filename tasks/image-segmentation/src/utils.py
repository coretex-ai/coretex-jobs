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
