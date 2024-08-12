from typing import Optional

import time
import logging
import subprocess

from coretex import currentTaskRun, folder_manager, CustomDataset

import requests
import ollama


OLLAMA_SERVER_URL = "http://127.0.0.1:11434"


def isOllamaInstalled() -> bool:
    try:
        subprocess.run(["ollama", "--version"], check = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def installOllama() -> None:
    try:
        subprocess.run("curl https://ollama.ai/install.sh | sh", shell = True, check = True)
        logging.info(">> [LLMTextProcessing] Ollama installation was successful")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f">> [LLMTextProcessing] An error occurred during Ollama installation: {e}")


def checkOllamaServer() -> bool:
    try:
        response = requests.get(OLLAMA_SERVER_URL)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.ConnectionError:
        return False


def launchOllamaServer() -> Optional[subprocess.Popen[bytes]]:
    if not isOllamaInstalled():
        installOllama()

    if checkOllamaServer():
        return None

    logging.info(">> [LLMTextProcessing] Launching ollama sever")
    return subprocess.Popen(["ollama", "serve"])


def pullModel(model: str, timeout: int = 60) -> None:
    serverOnline = checkOllamaServer()

    startTime = time.time()
    while not serverOnline and (time.time() - startTime) < timeout:
        time.sleep(1)
        serverOnline = checkOllamaServer()

    if not serverOnline:
        raise RuntimeError(">> [LLMTextProcessing] Failed to contact ollama server")

    ollama.pull(model)


def loadTextContent(dataset: CustomDataset) -> str:
    if dataset.count != 1:
        raise ValueError(">> [LLMTextProcessing] Dataset must have exactly one sample")

    dataset.download()
    sample = dataset.samples[0]
    sample.unzip()

    globbedTextFiles = list(sample.path.glob("*.txt"))
    if len(globbedTextFiles) != 1:
        raise ValueError(">> [LLMTextProcessing] Sample must contain a single .txt file")

    with open(globbedTextFiles[0], "r") as textFile:
        return textFile.read()


def main() -> None:
    taskRun = currentTaskRun()

    logging.info(">> [LLMTextProcessing] Loading text content from dataset")
    textContent = loadTextContent(taskRun.dataset)

    serverProcess = launchOllamaServer()

    try:
        prompt = taskRun.parameters["prompt"]
        model = taskRun.parameters["model"]

        logging.info(f">> [LLMTextProcessing] Downloading model: {model}")
        pullModel(model)

        logging.info(">> [LLMTextProcessing] Running inference on model")
        response = ollama.chat(
            model = model,
            messages = [{
                "role": "user",
                "content": f"Instruction: {prompt}\nText: {textContent}\""
            }]
        )

        logging.info(f">> [LLMTextProcessing] Prompt: {prompt}")

        responseContent = response["message"]["content"]
        logging.info(f">> [LLMTextProcessing] Response: {responseContent}")

        responseTextPath = folder_manager.temp / "response.txt"
        with responseTextPath.open('w', encoding = 'utf-8') as file:
            file.write(responseContent)

        if taskRun.createArtifact(responseTextPath, responseTextPath.name) is None:
            raise ValueError(">> [LLMTextProcessing] Failed to upload response as artifact")
    except Exception as e:
        logging.error(f">> [LLMTextProcessing] Error: {e}")
    finally:
        if serverProcess is not None:
            serverProcess.terminate()


if __name__ == "__main__":
    main()
