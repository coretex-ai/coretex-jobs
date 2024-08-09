from peft import LoraConfig
from transformers import TrainingArguments, BitsAndBytesConfig

import torch


def getPeftParameters(loraAlpha: float, loraDropout: float, rank: int) -> LoraConfig:
    return LoraConfig(
        lora_alpha = int(loraAlpha),
        lora_dropout = loraDropout,
        r = rank,
        bias = "none",
        task_type = "CAUSAL_LM"
    )


def getTrainingParameters(device: str, epochs: int, batchSize: int, learningRate: float, weightDecay: float) -> TrainingArguments:
    if device == "cuda":
        optimizer = "paged_adamw_32bit"

    if device == "cpu":
        optimizer = "adamw_torch"

    try:
        return TrainingArguments(
            output_dir = "./results_modified",
            num_train_epochs = epochs,
            per_device_train_batch_size = batchSize,
            per_device_eval_batch_size = batchSize,
            gradient_accumulation_steps = 1,
            optim = optimizer,
            save_steps = 25,
            logging_steps = 25,
            learning_rate = learningRate,
            weight_decay = weightDecay,
            fp16 = False,
            bf16 = False,
            max_grad_norm = 0.3,
            max_steps = -1,
            warmup_ratio = 0.03,
            group_by_length = True,
            lr_scheduler_type = "constant"
        )
    except Exception as e:
        raise Exception(f">> [Llama2Lora] Failed to instantiate training arguments. Error: {e}")


def getQuantizationConfig(float16: bool) -> BitsAndBytesConfig:
    dtype = torch.float16 if float16 else torch.float32

    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = dtype,
        bnb_4bit_use_double_quant = False
    )
