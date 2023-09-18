from pathlib import Path
from zipfile import ZipFile

import os
import logging

import tensorflowjs as tfjs
import tensorflow as tf
import coremltools

from coretex import Model, cache, Experiment, folder_manager, currentExperiment


classes = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


def fetchModelFile(modelUrl: str, fileName: str) -> str:
    if not cache.exists(modelUrl):
        cache.storeUrl(modelUrl, fileName)

    cachePath = cache.getPath(modelUrl)
    with ZipFile(cachePath, "r") as zipFile:
        zipFile.extractall(folder_manager.cache)

    return str(folder_manager.cache / Path(fileName).stem)


def saveTfLiteModel(savedModelPath: str) -> None:
    modelPath = folder_manager.temp / "model"
    converter = tf.lite.TFLiteConverter.from_saved_model(savedModelPath)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()
    with open(f"{modelPath}/model.tflite", 'wb') as f:
        f.write(tflite_model)


def saveCoremlModel(savedModelPath: str) -> None:
    modelPath = folder_manager.temp / "model"
    model = coremltools.converters.convert(savedModelPath)
    model.save(f"{modelPath}/model.mlmodel")


def saveJSModel(loadModelPath: str, modelPath: Path, modelTfjsPath: str, experiment: Experiment, coretexModel: Model) -> None:
    tfjs.converters.convert_tf_saved_model(
        loadModelPath,
        modelTfjsPath
    )

    coretexModel.saveModelDescriptor(modelPath, {
        "project_task": experiment.projectType,
        "labels": classes,
        "modelName": coretexModel.name,
        "description": experiment.description,

        "input_description": """
            The height/width must be multiple of 32.
            The height to width ratio should be close to cover the original image's aspect ratio.
            Optimal value for the larger side is 256 (this one can be adjusted based on the speed/accuracy requirements).
            For example, a 720p image (i.e. 720x1280 (HxW)) should be resized and padded to 160x256 image.
            Input is an array of 4 different values:
            Index 0: The first dimension is the batch dimension.
            Index 1: Height of image
            Index 2: Width of image
            Index 3: Number of channels in image

            Initial release of the tflite MoveNet.Multipose model.
            Note that the current model uses dynamic shape input which makes it incompatible with accelerators,
            e.g. GPU. We are working on the accelerator-compatible version and will release them soon.

            None/Null/Nil values mark dynamic (variable) amount of values
        """,
        "input_shape": [1, None, None, 3],

        "output_description": """
            Output is an array of 3 different values:
            Index 0: The first dimension is the batch dimension.
            Index 1: The second dimension corresponds to the maximum number of instance detections.
            The model can detect up to 6 people in the image frame simultaneously.
            Index 2: The third dimension represents the predicted bounding box/keypoint locations and scores.

            The first 17 * 3 elements are the keypoint locations and scores in the format: [y_0, x_0, s_0, y_1,
            x_1, s_1, …, y_16, x_16, s_16], where y_i, x_i, s_i are the yx-coordinates (normalized to image frame,
            e.g. range in [0.0, 1.0]) and confidence scores of the i-th joint correspondingly.

            The order of the 17 keypoint joints is: [nose, left eye, right eye, left ear, right ear, left shoulder,
            right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee,
            right knee, left ankle, right ankle]. The remaining 5 elements [ymin, xmin, ymax, xmax, score] represent
            the region of the bounding box (in normalized coordinates) and the confidence score of the instance.

        """,
        "output_shape": [1, 6, 56]
    })


def main() -> None:
    experiment: Experiment = currentExperiment()

    modelPath = folder_manager.createTempFolder("model")

    tfjsModelUrl = experiment.parameters["tfjsModelUrl"]
    tfjsFilename = "multipose_tfjs.zip"
    tfjsModelPath = os.path.join(modelPath, "tensorflowjs-model")

    savedModelUrl = experiment.parameters["savedModelUrl"]
    savedModelFilename = "multipose_savedModel.zip"

    coretexModel = Model.createModel(experiment.name, experiment.id, 0.9139, {})
    logging.info(f">> [Workspace] Model accuracy is: {coretexModel.accuracy}")

    savedModelPath = fetchModelFile(savedModelUrl, savedModelFilename)

    saveCoremlModel(savedModelPath)
    saveTfLiteModel(savedModelPath)
    saveJSModel(fetchModelFile(tfjsModelUrl, tfjsFilename), modelPath, tfjsModelPath, experiment, coretexModel)

    coretexModel.upload(modelPath)


if __name__ == "__main__":
    main()
