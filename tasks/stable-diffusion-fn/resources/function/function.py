from typing import Any, Dict
from base64 import encodebytes

import io
import logging

from coretex import functions
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import torch


MODEL_ID = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype = torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cpu")


def response(requestData: Dict[str, Any]) -> Dict[str, Any]:
    prompt = requestData.get("prompt")
    negativePrompt = requestData.get("negativePrompt")
    height = requestData.get("height")
    width = requestData.get("width")
    steps = requestData.get("steps")

    logging.debug(">> [StableDiffusion] Validating inputs")

    if not isinstance(prompt, str):
        return functions.badRequest("\"prompt\" has to be string")

    if negativePrompt is not None and not isinstance(negativePrompt, str):
        return functions.badRequest("\"negativePrompt\" has to be string or null")

    if not isinstance(height, int) and isinstance(width, int):
        height = width
    elif not isinstance(height, int):
        height = 768

    if not isinstance(width, int) and isinstance(height, int):
        width = height
    elif not isinstance(width, int):
        width = 768

    if not isinstance(steps, int):
        steps = 50

    if steps > 300:
        return functions.badRequest("Step limit of 300 exceeded")

    logging.debug(">> [StableDiffusion] Generating image")
    image = pipe(prompt, negative_prompt = negativePrompt, num_inference_steps = steps, width = width, height = height).images[0]

    byteArray = io.BytesIO()
    image.save(byteArray, "PNG")
    encodedImage = encodebytes(byteArray.getvalue()).decode("ascii")

    logging.debug(f">> [StableDiffusion] Image generated")

    return functions.success({"image": encodedImage})
