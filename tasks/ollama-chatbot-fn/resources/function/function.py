from typing import Any
from pathlib import Path

import uuid
import json
import logging

from coretex import functions, folder_manager

import ollama
import numpy as np

from utils import loadCorpusAndIndex, retrieveDocuments
from model import launchOllamaServer, pullModel, warmup, EMBEDDING_MODEL, LLM


memoryFolder = folder_manager.temp / "memories"
memoryFolder.mkdir(exist_ok = True)

indexDir = Path.cwd().parent / "corpus-index"
rag = indexDir.exists()

if rag:
    corpus, index = loadCorpusAndIndex(indexDir)


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    inputSessionId = requestData.get("session_id")
    sessionId = str(uuid.uuid1()) if inputSessionId is None else inputSessionId
    sessionPath =  memoryFolder / f"{sessionId}.json"

    query = requestData.get("query")
    if not isinstance(query, str):
        return functions.badRequest("Query cannot be empty")

    if inputSessionId is None or not sessionPath.exists():
        logging.debug(">>> Creating new session")
        if rag:
            queryEmbedding = ollama.embeddings(model = EMBEDDING_MODEL, prompt = query)["embedding"]
            retrievedDocuments = retrieveDocuments(
                np.array([queryEmbedding]),
                index,
                corpus,
                int(requestData.get("paragraphs", 5))
            )

            context = " ".join([doc[0] for doc in retrievedDocuments])
            suffix = f"Respond to the users queries with the help of the following data that has been retrieved through RAG: \n\n{context}\n\n Do not mention that this data was provided and just answer the queries."
        else:
            suffix = "Aid the user in any way you can."

        messages = [{
            "role": "system",
            "content": f"You are a helpful chatbot. {suffix}"
        }]
    else:
        with sessionPath.open("r") as file:
            messages: list[dict[str, str]] = json.load(file)  # type: ignore[no-redef]

    messages.append({
        "role": "user",
        "content": query
    })

    logging.debug(">>> Running inference on LLM")
    response = ollama.chat(
        model = LLM,
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

pullModel(LLM)
if rag:
    pullModel(EMBEDDING_MODEL)

warmup()
