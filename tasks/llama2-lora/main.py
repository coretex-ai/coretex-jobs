# Template is based on https://deci.ai/blog/fine-tune-llama-2-with-lora-for-question-answering/

import logging

from datasets import load_dataset
from trl import SFTTrainer
from transformers import pipeline, AutoTokenizer
from datasets import Dataset
from coretex import currentTaskRun, CustomDataset

from src.model import getModelName, loadModel, loadTokenizer, uploadModel
from src.configurations import getQuantizationConfig, getPeftParameters, getTrainingParameters


def loadData(dataset: CustomDataset) -> Dataset:
    dataset.download()
    for sample in dataset.samples:
        sample.unzip()

        ######                                        ######
        # Load data and perform preprocessing as necessary #
        ######                                        ######

        # Example for loading parquet files from a single sample
        dataFiles = list(sample.path.glob("*.parquet"))
        trainingData = load_dataset("parquet", data_files = [str(filePath)for filePath in dataFiles])["train"]

        return trainingData


def runInference(trainer: SFTTrainer, tokenizer: AutoTokenizer, prompt: str) -> str:
    textGenerator = pipeline(task = "text-generation", model = trainer.model, tokenizer = tokenizer, max_length = 200)
    output = textGenerator(f"<s>[INST] {prompt} [/INST]")
    return str(output[0]['generated_text'])


def main() -> None:
    taskRun = currentTaskRun()
    logging.info(">> [Llama2Lora] Loading dataset from coretex")
    trainingData = loadData(taskRun.dataset)

    modelName = getModelName(taskRun.parameters["modelVersion"])

    logging.info(">> [Llama2Lora] Loading Tokenizer")
    tokenizer = loadTokenizer(modelName, taskRun.parameters["context"])

    if taskRun.parameters["device"] == "cuda":
        quantConfig = getQuantizationConfig(taskRun.parameters["float16"])

    if taskRun.parameters["device"] == "cpu":  # bitsAndBytes quantization is not supported for CPU
        quantConfig = None

    peftParameters = getPeftParameters(
        taskRun.parameters["loraAlpha"],
        taskRun.parameters["loraDropout"],
        taskRun.parameters["loraRank"]
    )

    trainingParameters = getTrainingParameters(
        taskRun.parameters["device"],
        taskRun.parameters["epochs"],
        taskRun.parameters["batchSize"],
        taskRun.parameters["learningRate"],
        taskRun.parameters["weightDecay"]
    )

    logging.info(">> [Llama2Lora] Loading model")
    model = loadModel(modelName, taskRun.parameters["device"], quantConfig)

    trainer = SFTTrainer(
        model = model,
        train_dataset = trainingData,
        peft_config = peftParameters,
        dataset_text_field = "text",
        tokenizer = tokenizer,
        args = trainingParameters
    )

    logging.info(">> [Llama2Lora] Starting training")
    trainer.train()
    logging.info(">> [Llama2Lora] Training finished")

    ######                                 ######
    # Implement automatic evaluation as desired #
    accuracy = 1.0                              #
    ######                                 ######

    logging.info(">> [Llama2Lora] Uploading LoRA parameters to Coretex. You will need to load them together with the base model to perfrom inference")
    uploadModel(taskRun, trainer, accuracy)

    if taskRun.parameters["testPrompt"] is not None:
        response = runInference(trainer, tokenizer, taskRun.parameters["testPrompt"])
        logging.info(f">> [Llama2Lora] Test response: {response}")


if __name__ == "__main__":
    main()
