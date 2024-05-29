import requests


def downloadModelWeights(name: str, retryCount: int = 0) -> None:
    if retryCount >= 3:
        raise RuntimeError("Failed to download weights")

    url = f"https://github.com/THU-MIG/yolov10/releases/download/v1.1/{name}"
    with requests.get(url, stream = True) as r:
        if not r.ok:
            return downloadModelWeights(name, retryCount = retryCount + 1)

        with open(name, "wb") as file:
            for chunk in r.iter_content(chunk_size = 8192):
                file.write(chunk)
