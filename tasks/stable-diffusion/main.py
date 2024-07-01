from typing import Any, Optional
from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor

import logging
import uuid
import time

from coretex import currentTaskRun, TaskRun, ImageDataset, createDataset, folder_manager
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import torch


MODEL_ID = "stabilityai/stable-diffusion-2-1"


def loadModel(device: str) -> StableDiffusionPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype = dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


def getDefault(taskRun: TaskRun, name: str, default: Any) -> Any:
    value = taskRun.parameters.get(name)
    if value is None:
        return default

    return value


def generateImages(
    prompts: list[str],
    negativePrompt: Optional[str],
    width: int,
    height: int,
    steps: int,
    imageCount: int,
    seed: Optional[int]
) -> list[Path]:

    logging.info(">> [StableDiffsion] Loading model...")
    model = loadModel("cuda" if torch.cuda.is_available() else "cpu")

    prompts = prompts * imageCount
    logging.info(f">> [StableDiffusion] Generating {len(prompts)} images...")

    if negativePrompt is None:
        negativePrompts = None
    else:
        # Create an array equal to number of input prompts
        negativePrompts = [negativePrompt] * len(prompts)

    images = model(
        prompts,
        negative_prompt = negativePrompts,
        num_inference_steps = steps,
        width = width,
        height = height,
        seed = seed
    ).images

    imagePaths: list[Path] = []

    for image in images:
        imagePath = folder_manager.temp / f"{uuid.uuid4()}.jpg"
        image.save(imagePath)
        imagePaths.append(imagePath)

    return imagePaths


def uploadImage(taskRun: TaskRun, dataset: ImageDataset, imagePath: Path) -> None:
    try:
        dataset.add(imagePath)
        logging.info(f">> [StableDiffusion] Uploaded image \"{imagePath.stem}\"")
    except BaseException as ex:
        logging.error(f">> [StableDiffusion] Failed to upload image \"{imagePath.stem}\", reason: \"{ex}\"")

    try:
        artifact = taskRun.createArtifact(imagePath, imagePath.name)
        if artifact is None:
            logging.error(f">> [StableDiffusion] Failed to upload image \"{imagePath.stem}\" as artifact")
        else:
            logging.info(f">> [StableDiffusion] Uploaded image \"{imagePath.stem}\" as artifact")
    except BaseException as ex:
        logging.error(f">> [StableDiffusion] Failed to upload image \"{imagePath.stem}\" as artifact, reason: \"{ex}\"")


def main() -> None:
    taskRun = currentTaskRun()

    prompts: list[str] = taskRun.parameters["prompts"]
    negativePrompt: Optional[str] = taskRun.parameters.get("negativePrompts")
    width: int = taskRun.parameters["width"]
    height: int = taskRun.parameters["height"]
    imageCount: int = taskRun.parameters["imageCount"]
    seed: Optional[int] = taskRun.parameters.get("seed")
    steps: int = taskRun.parameters["steps"]

    if steps > 300:
        logging.warning(">> [StableDiffusion] Steps are larger than max value of 300, clipping...")
        steps = 300

    with ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor(max_workers = 8))
        dataset = stack.enter_context(createDataset(ImageDataset, f"stable-diffusion-{int(time.time())}", taskRun.projectId))

        for image in generateImages(prompts, negativePrompt, width, height, steps, imageCount, seed):
            executor.submit(uploadImage, taskRun, dataset, image)

        taskRun.submitOutput("generatedDataset", dataset)


if __name__ == "__main__":
    main()
