from dateutil import parser

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


modelVersion = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(modelVersion)
model = VisionEncoderDecoderModel.from_pretrained(modelVersion)


def trOCR(image: Image.Image) -> str:
    pixelValues = processor(image.convert("RGB"), return_tensors = "pt").pixel_values
    generatedIds = model.generate(pixelValues)
    return processor.batch_decode(generatedIds, skip_special_tokens = True)[0]  # type: ignore


def performOCR(images: list[Image.Image], classes: list[str]) -> dict[str, str]:
    detections: dict[str, str] = {}
    for i, image in enumerate(images):
        trOcrOutput = trOCR(image)

        if classes[i] == "date_of_birth":
            detections["date_of_birth_raw"] = trOcrOutput

            try:
                trOcrDateTime = parser.parse(trOcrOutput)
                trOcrOutputJson = {
                    "year": trOcrDateTime.year,
                    "month": trOcrDateTime.month,
                    "day": trOcrDateTime.day
                }
            except ValueError as e:
                trOcrOutputJson = {
                    "year": None,
                    "month": None,
                    "day": None
                }

            trOcrOutput = trOcrOutputJson

        detections[classes[i]] = trOcrOutput

    return detections
