from typing import Generator

import logging

from keras.layers import RandomFlip

import numpy as np
import tensorflow as tf

from coretex import TaskRun, TaskRunStatus, ImageDataset, folder_manager

from .utils import hasDotAnnotation


class Augment(tf.keras.layers.Layer):  # type: ignore[misc]

    def __init__(self, seed: int = 42) -> None:
        super().__init__()

        self.augmentInputs = RandomFlip(
            mode="horizontal",
            seed=seed
        )

        self.augmentLabels = RandomFlip(
            mode="horizontal",
            seed=seed
        )

    def call(self, inputs: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        inputs = self.augmentInputs(inputs)
        labels = self.augmentLabels(labels)

        return inputs, labels


def loadDataset(coretexDataset: ImageDataset, coretexTaskRun: TaskRun) -> tuple[int, tf.data.Dataset]:
    coretexTaskRun.updateStatus(TaskRunStatus.inProgress, "Created the TF dataset.")

    def datasetElementProvider() -> Generator:
        for sample in coretexDataset.samples:
            sample.unzip()

            data = sample.load()
            if data.annotation is None:
                raise ValueError

            if hasDotAnnotation(data.annotation):
                logging.warning(f">> [Image Segmentation] Sample \"{sample.name}\" (ID: {sample.id}) has invalid annotation (too few coordinates). Skipping Sample")
                continue

            segmentationMask = data.annotation.extractSegmentationMask(coretexDataset.classes)
            segmentationMask = np.expand_dims(segmentationMask, axis=-1)

            yield {
                "image": tf.convert_to_tensor(data.image, dtype = tf.uint8),
                "segmentation_mask": tf.convert_to_tensor(segmentationMask, dtype = tf.uint8)
            }

    dataset = tf.data.Dataset.from_generator(
        generator=datasetElementProvider,
        output_signature={
            "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            "segmentation_mask": tf.TensorSpec(shape=(None, None, 1), dtype=tf.uint8)
        }
    )

    return coretexDataset.count, dataset


def createBatches(
    dataset: tf.data.Dataset,
    count: int,
    validationSplit: float,
    batchSize: int,
    imageSize: int
) -> tuple[int, tf.data.Dataset, int, tf.data.Dataset]:

    def normalize(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        inputImage = tf.cast(image, tf.float32) / 255.0
        inputMask = tf.cast(mask, tf.uint8)

        return inputImage, inputMask

    def loadImage(element: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        inputImage = tf.image.resize_with_pad(element['image'], imageSize, imageSize)
        inputMask = tf.image.resize_with_pad(element['segmentation_mask'], imageSize, imageSize)

        return normalize(inputImage, inputMask)

    assert 0 <= validationSplit and validationSplit <= 1

    trainCount = int((1 - validationSplit) * count)
    testCount = count - trainCount

    trainImages = dataset.take(trainCount).map(
        loadImage,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    testImages = dataset.skip(trainCount).take(testCount).map(
        loadImage,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    trainBatches = (
        trainImages
        .cache(filename = str(folder_manager.temp / "training"))
        .shuffle(trainCount)
        .batch(batchSize)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size = tf.data.AUTOTUNE)
    )
    testBatches = testImages.batch(batchSize).repeat()

    return trainCount, trainBatches, testCount, testBatches
