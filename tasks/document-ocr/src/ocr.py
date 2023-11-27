from pathlib import Path

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from coretex import TaskRun

import pytesseract
import easyocr
import numpy as np

from .utils import createArtifact


# EasyOCR
reader = easyocr.Reader(["en"])
# TrOCR
modelVersion = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(modelVersion)
model = VisionEncoderDecoderModel.from_pretrained(modelVersion)


def trOCR(image: Image.Image) -> str:
    pixelValues = processor(image.convert("RGB"), return_tensors = "pt").pixel_values
    generatedIds = model.generate(pixelValues)
    return processor.batch_decode(generatedIds, skip_special_tokens = True)[0]  # type: ignore


def performOCR(images: list[Image.Image], classes: list[str], outputDir: Path, taskRun: TaskRun):
    for i, image in enumerate(images):
        classResultPath = outputDir / classes[i]
        classResultPath.mkdir(exist_ok = True)

        # tesseractOutputPath = classResultPath / "pytesseract.txt"
        # with tesseractOutputPath.open("w") as file:
        #     file.write(pytesseract.image_to_string(image))
        # createArtifact(taskRun, tesseractOutputPath, str(tesseractOutputPath.relative_to(outputDir.parent)))

        # easyOcrOutputPath = classResultPath / "easyocr.txt"
        # with easyOcrOutputPath.open("w") as file:
        #     result = reader.readtext(np.array(image))
        #     if len(result) > 0:
        #         file.write("\n".join([e[1] for e in result]))
        # createArtifact(taskRun, easyOcrOutputPath, str(easyOcrOutputPath.relative_to(outputDir.parent)))

        trOcrOutputPath = classResultPath.joinpath("trocr.txt")
        with trOcrOutputPath.open("w") as file:
            file.write(trOCR(image))
        createArtifact(taskRun, trOcrOutputPath, str(trOcrOutputPath.relative_to(outputDir.parent)))
