from typing import Any
from pathlib import Path

import uuid
import json
import logging

from coretex import functions, folder_manager

import ollama

from model import launchOllamaServer, pullModel, warmup

configPath = Path.cwd().parent / "config.json"
if configPath.exists():
    with configPath.open("r") as file:
        model = json.load(file)["model"]
else:
    logging.error("No config file found in the endpoint repository. Using llama3")
    model = "llama3"


memoryFolder = folder_manager.temp / "memories"
memoryFolder.mkdir(exist_ok = True)


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    inputSessionId = requestData.get("session_id")

    if inputSessionId is None:
        sessionId = str(uuid.uuid1())
    else:
        sessionId = inputSessionId

    sessionPath =  memoryFolder / f"{sessionId}.json"

    query = requestData.get("query")
    if query == None:
        functions.badRequest("Query cannot be empty")

    if inputSessionId is None or not sessionPath.exists():
        logging.debug(">>> Creating new session")
        messages = [{
            "role": "system",
            "content": f"You are an agent of Coretex. You will talk with \"user\". Assist, converse, find. The conversation must remain normal."
        }]
    else:
        with sessionPath.open("r") as file:
            messages: list[dict[str, str]] = json.load(file)

    messages.append({
        "role": "user",
        "content": query
    })

    logging.debug(">>> Running inference on LLM")
    response = ollama.chat(
        model = model,
        messages = messages
    )

    messages.append(response["message"])

    with sessionPath.open("w") as file:
        json.dump(messages, file)

    answer = response["message"]["content"]
    logging.debug(f">>> Returning response:\n{answer}")

    return functions.success({
        "response": answer,
        "sessionId": sessionId
    })


launchOllamaServer()
pullModel(model)
warmup(model)
