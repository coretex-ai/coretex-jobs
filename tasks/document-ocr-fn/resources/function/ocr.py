from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import pytesseract
import easyocr
import numpy as np


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


def performOCR(images: list[Image.Image], classes: list[str]) -> list[dict[str, str]]:
    detections: list[dict[str, str]] = []
    for i, image in enumerate(images):
        # tesseractOutput = pytesseract.image_to_string(image).replace("\n", " ")
        # easyOcrOutput = " ".join([e[1] for e in reader.readtext(np.array(image))])
        trOcrOutput = trOCR(image)

        detections.append({
            "class": classes[i],
            # "tesseract": tesseractOutput,
            # "easyOCR": easyOcrOutput,
            "trOCR": trOcrOutput
        })

    return detections
