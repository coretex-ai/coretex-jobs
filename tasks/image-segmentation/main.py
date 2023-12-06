from pathlib import Path

import logging
import math

from keras import Model as KerasModel
from keras.callbacks import History

import tensorflow as tf
import tensorflowjs as tfjs
import coremltools

from coretex import TaskRunStatus, Model, ImageSegmentationDataset, TaskRun, Metric, MetricType, currentTaskRun, folder_manager

from src import detect
from src.model import UNetModel
from src.dataset import loadDataset, createBatches
from src.callbacks import DisplayCallback


def saveLiteModel(model: KerasModel, path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tfLiteModel = converter.convert()

    with path.open("wb") as file:
        file.write(tfLiteModel)


def saveCoremlModel(model: KerasModel, path: Path) -> None:
    model = coremltools.converters.convert(model)
    model.save(str(path))


def saveTFJSModel(modelPath: Path, path: Path) -> None:
    tfjs.converters.convert_tf_saved_model(
        str(modelPath),
        str(path)
    )


def saveModelDescriptor(
    model: KerasModel,
    taskRun: TaskRun[ImageSegmentationDataset],
    coretexModel: Model,
    path: Path
) -> None:

    labels = [
        {
            "label": clazz.label,
            "color": clazz.color
        }
        for clazz in taskRun.dataset.classes
    ]

    coretexModel.saveModelDescriptor(path, {
        "project_task": taskRun.projectType,
        "labels": labels,
        "modelName": coretexModel.name,
        "description": taskRun.description,

        "input_description": "RGB image",
        "input_shape": model.input_shape,

        "output_description": "Segmentation mask",
        "output_shape": model.output_shape
    })


def main() -> None:
    taskRun: TaskRun[ImageSegmentationDataset] = currentTaskRun()

    taskRun.createMetrics([
        Metric.create("loss", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]]),
        Metric.create("accuracy", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]], [0, 1]),
        Metric.create("val_loss", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]]),
        Metric.create("val_accuracy", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]], [0, 1])
    ])

    taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset")
    taskRun.dataset.download()

    excludedClasses: list[str] = taskRun.parameters["excludedClasses"]
    logging.info(f">> [Image Segmentation] Excluding classes: {excludedClasses}")
    taskRun.dataset.classes.exclude(excludedClasses)

    count, dataset = loadDataset(taskRun.dataset, taskRun)
    trainCount, trainBatches, testCount, testBatches = createBatches(
        dataset,
        count,
        taskRun.parameters["validationSplit"],
        taskRun.parameters["batchSize"],
        taskRun.parameters["imageSize"]
    )

    # + 1 because we also have a background class
    classCount = len(taskRun.dataset.classes) + 1
    model = UNetModel(classCount, taskRun.parameters["imageSize"])

    taskRun.updateStatus(TaskRunStatus.inProgress, "Training the model")

    epochs: int = taskRun.parameters["epochs"]

    history: History = model.fit(
        trainBatches,
        epochs = epochs,
        steps_per_epoch = math.ceil(trainCount / taskRun.parameters["batchSize"]),
        validation_steps = math.ceil(testCount / taskRun.parameters["batchSize"]),
        validation_data = testBatches,
        callbacks = [DisplayCallback(epochs)],
        verbose = 0,
        use_multiprocessing = True
    )

    detect.run(taskRun, model, taskRun.dataset)

    coretexModel = Model.createModel(
        taskRun.name,
        taskRun.id,
        history.history["val_accuracy"][-1],  # gets the validation accuracy after last epoch
        {}
    )

    logging.info(f">> [Image Segmentation] Model accuracy is: {coretexModel.accuracy}")

    modelDirPath = folder_manager.createTempFolder("model")
    model.save(modelDirPath / "tensorflow-model")

    logging.info(">> [Image Segmentation] Converting model to TFLite format")
    saveLiteModel(model, modelDirPath / "model.tflite")

    logging.info(">> [Image Segmentation] Converting model to CoreML format")
    saveCoremlModel(model, modelDirPath / "model.mlmodel")

    logging.info(">> [Image Segmentation] Converting model to TFJS format")
    saveTFJSModel(modelDirPath / "tensorflow-model", modelDirPath / "tensorflowjs-model")

    logging.info(">> [Image Segmentation] Saving model descriptor")
    saveModelDescriptor(model, taskRun, coretexModel, modelDirPath)

    coretexModel.upload(folder_manager.temp / "model")


if __name__ == "__main__":
    main()
