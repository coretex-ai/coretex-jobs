from typing import Optional

import logging
import subprocess

from ollama import ResponseError

import time
import requests
import ollama


OLLAMA_SERVER_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM = "llama3"


def warmup() -> None:
    ollama.chat(
    model = LLM,
    messages = [{
        "role": "user",
        "content": f"This is a test and a warmup. Respond with \"OK\"."
    }]
)


def isOllamaInstalled() -> bool:
    try:
        subprocess.run(["ollama", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def installOllama() -> None:
    try:
        subprocess.run("curl https://ollama.ai/install.sh | sh", shell=True, check=True)
        logging.info(">> [OllamaRAG] Ollama installation was successful")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f">> [OllamaRAG] An error occurred during Ollama installation: {e}")


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

    logging.info(">> [OllamaRAG] Launching ollama sever")
    return subprocess.Popen(["ollama", "serve"])


def pullModel(model: str) -> None:
    # Sleep for 1 second to give the server time to start
    time.sleep(1)

    try:
        ollama.pull(model)
    except ResponseError:
        pullModel(model)
