from typing import Any, Optional
from pathlib import Path

from coretex import folder_manager
from coretex.networking import success, badRequest

import detect


def prepareInputData(inputPath: Path) -> Optional[Path]:
    imageDir = folder_manager.createTempFolder("images")

    if isinstance(inputPath, Path) and inputPath.is_file():
        inputPath.link_to(imageDir / inputPath.name)
        return imageDir

    return None


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    modelDir = requestData.get("model")
    weightsPath = modelDir / "model.pt"

    imageSize = requestData.get("imageSize", 640)
    if isinstance(imageSize, str):
        imageSize = int(imageSize)

    inputPath = prepareInputData(requestData.get("inputFile"))
    if inputPath is None:
        return badRequest("Invalid input data. Single image file expected")

    boundingBoxes, classes = detect.run(inputPath, weights = weightsPath, imgsz = (imageSize, imageSize))

    return success({
        "bbox": boundingBoxes,
        "class": classes
    })
