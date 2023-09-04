import logging
import os
import shutil

from keras import Model as KerasModel
from keras.callbacks import History

import tensorflow as tf
import tensorflowjs as tfjs
import coremltools

from coretex import RunStatus, Model, ImageSegmentationDataset, ExecutingRun, Metric, MetricType
from coretex.job import initializeJob
from coretex.folder_management import FolderManager

from src.model import UNetModel
from src.dataset import loadDataset, createBatches
from src.callbacks import DisplayCallback
from src.utils import saveDatasetPredictions


def saveLiteModel(model: KerasModel):
    modelPath = FolderManager.instance().getTempFolder("model")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(f"{modelPath}/model.tflite", 'wb') as f:
        f.write(tflite_model)


def saveCoremlModel(model: KerasModel):
    modelPath = FolderManager.instance().getTempFolder("model")
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


def saveJSModel(model: KerasModel, run: ExecutingRun[ImageSegmentationDataset], coretexModel: Model):
    modelPath = FolderManager.instance().getTempFolder("model")
    saveTFJSModelFromTFModel(model, modelPath)

    labels = [
        {
            "label": clazz.label,
            "color": clazz.color
        }
        for clazz in run.dataset.classes
    ]

    coretexModel.saveModelDescriptor(modelPath, {
        "project_task": run.spaceTask,
        "labels": labels,
        "modelName": coretexModel.name,
        "description": run.description,

        "input_description": "RGB image",
        "input_shape": model.input_shape,

        "output_description": "Segmentation mask",
        "output_shape": model.output_shape
    })


def main(run: ExecutingRun[ImageSegmentationDataset]):
    run.createMetrics([
        Metric.create("loss", "epoch", MetricType.int, "value", MetricType.float, [0, run.parameters["epochs"]]),
        Metric.create("accuracy", "epoch", MetricType.int, "value", MetricType.float, [0, run.parameters["epochs"]], [0, 1])
    ])

    # path to which the model will be saved after training
    FolderManager.instance().createTempFolder("model")

    run.updateStatus(RunStatus.inProgress, "Downloading dataset")
    run.dataset.download()

    excludedClasses: list[str] = run.parameters["excludedClasses"]
    logging.info(f">> [Workspace] Excluding classes: {excludedClasses}")
    run.dataset.classes.exclude(excludedClasses)

    count, dataset = loadDataset(run.dataset, run)
    trainCount, trainBatches, testCount, testBatches = createBatches(
        dataset,
        count,
        run.parameters["validationSplit"],
        run.parameters["bufferSize"],
        run.parameters["batchSize"],
        run.parameters["imageSize"]
    )

    # + 1 because we also have a background class
    classCount = len(run.dataset.classes) + 1
    model = UNetModel(classCount, run.parameters["imageSize"])

    sample = testBatches.take(1).take(1)
    saveDatasetPredictions("BeforeTraining", model, sample)

    run.updateStatus(RunStatus.inProgress, "Training the model")

    epochs: int = run.parameters["epochs"]
    history: History = model.fit(
        trainBatches,
        epochs = epochs,
        steps_per_epoch = trainCount // run.parameters["batchSize"],
        validation_steps = testCount // run.parameters["batchSize"] // run.parameters["validationSubSplits"],
        validation_data = testBatches,
        callbacks = [DisplayCallback(model, sample, epochs)],
        verbose = 0,
        use_multiprocessing = True
    )

    # Runs prediction for all test data and uploads it to coretex as artifacts
    saveDatasetPredictions("AfterTraining", model, testBatches)

    run.updateStatus(RunStatus.inProgress, "Postprocessing model")

    coretexModel = Model.createModel(
        run.name,
        run.id,
        history.history["accuracy"][-1],  # gets the accuracy after last epoch
        {}
    )

    logging.info(f">> [Workspace] Model accuracy is: {coretexModel.accuracy}")

    saveLiteModel(model)
    saveCoremlModel(model)
    saveJSModel(model, run, coretexModel)

    coretexModel.upload(FolderManager.instance().getTempFolder("model"))


if __name__ == "__main__":
    initializeJob(main)
