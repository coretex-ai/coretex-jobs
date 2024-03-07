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
        functions.badRequest("\"prompt\" has to be string")

    if negativePrompt is not None and not isinstance(negativePrompt, str):
        functions.badRequest("\"negativePrompt\" has to be string or null")

    if isinstance(height, str) and height.isdecimal():
        height = int(height)
    elif height is None:
        height = 768
    else:
        return functions.badRequest("\"height\" has to be an integer")

    if isinstance(width, str) and width.isdecimal():
        width = int(width)
    elif width is None:
        width = 768
    else:
        return functions.badRequest("\"width\" has to be an integer")

    if isinstance(steps, str) and steps.isdecimal():
        steps = int(steps)
        if steps > 300:
            return functions.badRequest("Step limit of 300 exceeded")
    elif steps is None:
        steps = 50
    else:
        return functions.badRequest("\"steps\" has to be an integer")

    logging.debug(">> [StableDiffusion] Generating image")
    image = pipe(prompt, negative_prompt = negativePrompt, num_inference_steps = steps, width = 512, height = 512).images[0]

    byteArray = io.BytesIO()
    image.save(byteArray, "PNG")
    encodedImage = encodebytes(byteArray.getvalue()).decode("ascii")

    logging.debug(f">> [StableDiffusion] Image generated")

    return functions.success({"image": encodedImage})
