import logging
import os
import shutil

from keras import Model as KerasModel
from keras.callbacks import History

import tensorflow as tf
import tensorflowjs as tfjs
import coremltools

from coretex import ExperimentStatus, Model, ImageSegmentationDataset, Experiment, Metric, MetricType, currentExperiment, folder_manager

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


def saveJSModel(model: KerasModel, experiment: Experiment[ImageSegmentationDataset], coretexModel: Model):
    modelPath = folder_manager.temp / "model"
    saveTFJSModelFromTFModel(model, modelPath)

    labels = [
        {
            "label": clazz.label,
            "color": clazz.color
        }
        for clazz in experiment.dataset.classes
    ]

    coretexModel.saveModelDescriptor(modelPath, {
        "project_task": experiment.projectType,
        "labels": labels,
        "modelName": coretexModel.name,
        "description": experiment.description,

        "input_description": "RGB image",
        "input_shape": model.input_shape,

        "output_description": "Segmentation mask",
        "output_shape": model.output_shape
    })


def main() -> None:
    experiment: Experiment[ImageSegmentationDataset] = currentExperiment()

    experiment.createMetrics([
        Metric.create("loss", "epoch", MetricType.int, "value", MetricType.float, [0, experiment.parameters["epochs"]]),
        Metric.create("accuracy", "epoch", MetricType.int, "value", MetricType.float, [0, experiment.parameters["epochs"]], [0, 1])
    ])

    # path to which the model will be saved after training
    folder_manager.createTempFolder("model")

    experiment.updateStatus(ExperimentStatus.inProgress, "Downloading dataset")
    experiment.dataset.download()

    excludedClasses: list[str] = experiment.parameters["excludedClasses"]
    logging.info(f">> [Workspace] Excluding classes: {excludedClasses}")
    experiment.dataset.classes.exclude(excludedClasses)

    count, dataset = loadDataset(experiment.dataset, experiment)
    trainCount, trainBatches, testCount, testBatches = createBatches(
        dataset,
        count,
        experiment.parameters["validationSplit"],
        experiment.parameters["bufferSize"],
        experiment.parameters["batchSize"],
        experiment.parameters["imageSize"]
    )

    # + 1 because we also have a background class
    classCount = len(experiment.dataset.classes) + 1
    model = UNetModel(classCount, experiment.parameters["imageSize"])

    sample = testBatches.take(1).take(1)
    saveDatasetPredictions("BeforeTraining", model, sample)

    experiment.updateStatus(ExperimentStatus.inProgress, "Training the model")

    epochs: int = experiment.parameters["epochs"]
    history: History = model.fit(
        trainBatches,
        epochs = epochs,
        steps_per_epoch = trainCount // experiment.parameters["batchSize"],
        validation_steps = testCount // experiment.parameters["batchSize"] // experiment.parameters["validationSubSplits"],
        validation_data = testBatches,
        callbacks = [DisplayCallback(model, sample, epochs)],
        verbose = 0,
        use_multiprocessing = True
    )

    # Runs prediction for all test data and uploads it to coretex as artifacts
    saveDatasetPredictions("AfterTraining", model, testBatches)

    experiment.updateStatus(ExperimentStatus.inProgress, "Postprocessing model")

    coretexModel = Model.createModel(
        experiment.name,
        experiment.id,
        history.history["accuracy"][-1],  # gets the accuracy after last epoch
        {}
    )

    logging.info(f">> [Workspace] Model accuracy is: {coretexModel.accuracy}")

    saveLiteModel(model)
    saveCoremlModel(model)
    saveJSModel(model, experiment, coretexModel)

    coretexModel.upload(folder_manager.temp / "model")


if __name__ == "__main__":
    main()
