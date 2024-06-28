from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from coretex import Model, TaskRun, folder_manager

import torch


def getModelName(modelVersion: str) -> str:
    if modelVersion == "7b-chat":
        return "NousResearch/Llama-2-7b-chat-hf"

    if modelVersion == "13b-chat":
        return "NousResearch/Llama-2-13b-chat-hf"

    if modelVersion == "70b-chat":
        return "NousResearch/Llama-2-70b-chat-hf"

    raise ValueError("Invalid model version")


def loadTokenizer(modelName:  str, context: Optional[int] = None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True, model_max_length = context)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def loadModel(modelName: str, device: str, quantConfig: Optional[BitsAndBytesConfig]) -> AutoModelForCausalLM:
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(">> [Llama2Lora] Device parameter is cuda, but cuda is not available")

    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        quantization_config = quantConfig,
        device_map = device
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model


def uploadModel(taskRun: TaskRun, trainer: SFTTrainer, accuracy: float) -> None:
    modelVersion = taskRun.parameters["modelVersion"]
    modelFolder = folder_manager.createTempFolder("model")
    newModelPath = modelFolder / f"{taskRun.id}-fine-tuned-llama2-{modelVersion}-lora-adapters"

    trainer.model.save_pretrained(newModelPath)
    coretexModel = Model.createModel(newModelPath.name, taskRun.id, accuracy, {})
    coretexModel.upload(modelFolder)

    taskRun.submitOutput("outputModel", coretexModel)
