from typing import Any, Optional

import time
import logging
import subprocess

from coretex import functions
from ollama import ResponseError

import ollama
import requests


OLLAMA_SERVER_URL = "http://127.0.0.1:11434"
TEXT_PROMPT_PREFIX = "Grade the sentiment of the following text: "
SYSTEM_PROMPT = [
    {
        "role": "user",
        "content": "You are a sentiment analysis bot. You will recieve a text which you will grade as one of five sentiment categories ([very positive], [positive], [neutral], [negative], [very negative]). Your respose will follow this format \"Sentiment: positive\". Is this understood?"
    },
    {
        "role": "assistant",
        "content": "Understood. All following responses will follow format \"Sentiment: category\""
    },
    {
        "role": "user",
        "content": f"{TEXT_PROMPT_PREFIX}\"Absolutely in love with this new product! The quality is unbelievable, and it's so advanced. Highly recommend to any humans out there!\""
    },
    {
        "role": "assistant",
        "content": "Sentiment: very positive"
    }
]


def isOllamaInstalled() -> bool:
    try:
        subprocess.run(["ollama", "--version"], check = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def installOllama() -> None:
    try:
        subprocess.run("curl https://ollama.ai/install.sh | sh", shell = True, check = True)
        logging.info(">> [OSentimentAnalysis] Ollama installation was successful")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f">> [OSentimentAnalysis] An error occurred during Ollama installation: {e}")


def checkOllamaServer() -> Optional[bool]:
    try:
        response = requests.get(OLLAMA_SERVER_URL)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def launchOllamaServer() -> subprocess.Popen[bytes]:
    if not isOllamaInstalled():
        installOllama()

    if checkOllamaServer():
        return None

    logging.info(">> [OSentimentAnalysis] Launching ollama sever")
    return subprocess.Popen(["ollama", "serve"])


def pullModel() -> None:
    # Sleep for 1 second to give the server time to start
    time.sleep(1)

    try:
        ollama.pull("llama2")
    except ResponseError:
        pullModel()


# Available roles: user, assistant
def response(requestData: dict[str, Any]) -> dict[str, Any]:
    text = requestData.get("text")
    if text is None:
        return functions.badRequest("No \"text\" field in input")

    logging.info("Running inference")
    response = ollama.chat(
        model = "llama2",
        messages = SYSTEM_PROMPT + [{
            "role": "user",
            "content": f"{TEXT_PROMPT_PREFIX}\"{text}\""
        }]
    )

    try:
        logging.info("Retrieving sentiment")
        sentiment = response["message"]["content"].split(sep = ": ")[1]
    except Exception as e:
        return {
            "code": 500,
            "body": {
                "error": f">> [OSentimentAnalysis] Failed to retrieve sentiment from ollama response. Error: {e}"
            }
        }

    logging.info(f"Returning sentiment \"{sentiment}\"")
    return functions.success({"sentiment": sentiment})


launchOllamaServer()
pullModel()
