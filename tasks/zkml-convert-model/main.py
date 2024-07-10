from pathlib import Path

import json
import asyncio
import logging

from coretex import currentTaskRun, folder_manager, Model

import ezkl
import onnx
import numpy as np


def generateDummyInput(onnxModelPath: Path) -> dict[str, list[float]]:
    model = onnx.load(onnxModelPath)

    inputTensor = model.graph.input[0]
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[inputTensor.type.tensor_type.elem_type]
    shape = [dim.dim_value for dim in inputTensor.type.tensor_type.shape.dim]
    shape[0] = 1

    data = np.random.random(shape).astype(dtype)
    flattenedData = np.array(data).reshape(-1).tolist()

    return dict(input_data = [flattenedData])


async def compileModel(
    onnxPath: Path,
    settings: Path,
    compiledModel: Path,
    verifKey: Path,
    proofKey: Path
) -> None:

    inputPath = folder_manager.temp / "input.json"
    with inputPath.open("w") as file:
        json.dump(generateDummyInput(onnxPath), file)

    await ezkl.calibrate_settings(inputPath, onnxPath, settings, target = "resources", max_logrows = 12, scales = [2])

    ezkl.compile_circuit(onnxPath, compiledModel, settings)
    await ezkl.get_srs(settings)
    ezkl.setup(compiledModel, verifKey, proofKey)


def main() -> None:
    taskRun = currentTaskRun()

    ctxOnnxModel: Model = taskRun.parameters["onnxModel"]
    ctxOnnxModel.download()

    modelDir = folder_manager.createTempFolder("model")

    onnxPaths = list(ctxOnnxModel.path.glob("*.onnx"))
    if len(onnxPaths) != 1:
        raise ValueError(f">> [ZKLMConversion] Model files have to contain exactly one .onnx file. Found {len(onnxPaths)}")

    onnxPath = onnxPaths[0]

    try:
        onnxPath.link_to(modelDir / onnxPath.name)
    except AttributeError:
        modelDir.joinpath(onnxPath.name).hardlink_to(onnxPath)  # type: ignore[attr-defined]

    # Define paths
    compiledModelPath = modelDir / "model.compiled"
    proofKey = modelDir / "prove.pk"
    verifKey = modelDir / "verify.pk"
    settingsPath = modelDir / "settings.json"

    logging.info(">> [ZKLMConversion] Setting up EZKL")

    visibilities = [
        taskRun.parameters["inputVisibility"],
        taskRun.parameters["outputVisibility"],
        taskRun.parameters["modelVisibility"]
    ]

    if visibilities.count(False) > 1:
        raise ValueError("[ZKLMConversion] Only one of three visibility parameters can be private (False)")

    pyRunArgs = ezkl.PyRunArgs()
    pyRunArgs.input_visibility = "public" if visibilities[0] else "private"
    pyRunArgs.output_visibility = "public" if visibilities[1] else "private"
    pyRunArgs.param_visibility = "fixed" if visibilities[2] else "private"

    logging.info(">> [ZKLMConversion] Generating settings")
    ezkl.gen_settings(onnxPath, settingsPath, py_run_args = pyRunArgs)

    logging.info(">> [ZKLMConversion] Compiling model")
    asyncio.run(compileModel(onnxPath, settingsPath, compiledModelPath, verifKey, proofKey))

    logging.info(">> [ZKLMConversion] EZKL setup complete")
    logging.info(">> [ZKLMConversion] Uploading model and EZKL files")
    ctxModel = Model.createModel(taskRun.generateEntityName(), taskRun.projectId, ctxOnnxModel.accuracy, {})
    ctxModel.upload(modelDir)


if __name__ == "__main__":
    main()
