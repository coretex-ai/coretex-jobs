import logging
import os
import shutil

from keras import Model as KerasModel
from keras.callbacks import History

import tensorflow as tf
import tensorflowjs as tfjs
import coremltools

from coretex import TaskRunStatus, Model, ImageSegmentationDataset, TaskRun, Metric, MetricType, currentTaskRun, folder_manager

from src.model import UNetModel
from src.dataset import loadDataset, createBatches
from src.callbacks import DisplayCallback
from src.utils import saveDatasetPredictions


def saveLiteModel(model: KerasModel):
    modelPath = folder_manager.temp / "model"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(f"{modelPath}/model.tflite", 'wb') as f:
        f.write(tflite_model)


def saveCoremlModel(model: KerasModel):
    modelPath = folder_manager.temp / "model"
    model = coremltools.converters.convert(model)
    model.save(f"{modelPath}/model.mlmodel")


def saveTFJSModelFromTFModel(model: KerasModel, path: str) -> None:
    tensorflowModelPath = os.path.join(path, "tensorflow-model")
    model.save(tensorflowModelPath)

    tensorflowJSModelPath = os.path.join(path, "tensorflowjs-model")
    tfjs.converters.convert_tf_saved_model(
        tensorflowModelPath,
        tensorflowJSModelPath
    )

    shutil.rmtree(tensorflowModelPath)


def saveJSModel(model: KerasModel, taskRun: TaskRun[ImageSegmentationDataset], coretexModel: Model):
    modelPath = folder_manager.temp / "model"
    saveTFJSModelFromTFModel(model, modelPath)

    labels = [
        {
            "label": clazz.label,
            "color": clazz.color
        }
        for clazz in taskRun.dataset.classes
    ]

    coretexModel.saveModelDescriptor(modelPath, {
        "project_task": taskRun.spaceTask,
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
        Metric.create("accuracy", "epoch", MetricType.int, "value", MetricType.float, [0, taskRun.parameters["epochs"]], [0, 1])
    ])

    # path to which the model will be saved after training
    folder_manager.createTempFolder("model")

    taskRun.updateStatus(TaskRunStatus.inProgress, "Downloading dataset")
    taskRun.dataset.download()

    excludedClasses: list[str] = taskRun.parameters["excludedClasses"]
    logging.info(f">> [Workspace] Excluding classes: {excludedClasses}")
    taskRun.dataset.classes.exclude(excludedClasses)

    count, dataset = loadDataset(taskRun.dataset, taskRun)
    trainCount, trainBatches, testCount, testBatches = createBatches(
        dataset,
        count,
        taskRun.parameters["validationSplit"],
        taskRun.parameters["bufferSize"],
        taskRun.parameters["batchSize"],
        taskRun.parameters["imageSize"]
    )

    # + 1 because we also have a background class
    classCount = len(taskRun.dataset.classes) + 1
    model = UNetModel(classCount, taskRun.parameters["imageSize"])

    sample = testBatches.take(1).take(1)
    saveDatasetPredictions("BeforeTraining", model, sample)

    taskRun.updateStatus(TaskRunStatus.inProgress, "Training the model")

    epochs: int = taskRun.parameters["epochs"]
    history: History = model.fit(
        trainBatches,
        epochs = epochs,
        steps_per_epoch = trainCount // taskRun.parameters["batchSize"],
        validation_steps = testCount // taskRun.parameters["batchSize"] // taskRun.parameters["validationSubSplits"],
        validation_data = testBatches,
        callbacks = [DisplayCallback(model, sample, epochs)],
        verbose = 0,
        use_multiprocessing = True
    )

    # Runs prediction for all test data and uploads it to coretex as artifacts
    saveDatasetPredictions("AfterTraining", model, testBatches)

    taskRun.updateStatus(TaskRunStatus.inProgress, "Postprocessing model")

    coretexModel = Model.createModel(
        taskRun.name,
        taskRun.id,
        history.history["accuracy"][-1],  # gets the accuracy after last epoch
        {}
    )

    logging.info(f">> [Workspace] Model accuracy is: {coretexModel.accuracy}")

    saveLiteModel(model)
    saveCoremlModel(model)
    saveJSModel(model, taskRun, coretexModel)

    coretexModel.upload(folder_manager.temp / "model")


if __name__ == "__main__":
    main()
