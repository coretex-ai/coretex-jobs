from typing import Optional
from dateutil import parser
from datetime import datetime

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


modelVersion = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(modelVersion)
model = VisionEncoderDecoderModel.from_pretrained(modelVersion)


def trOCR(image: Image.Image) -> str:
    pixelValues = processor(image.convert("RGB"), return_tensors = "pt").pixel_values
    generatedIds = model.generate(pixelValues)
    return processor.batch_decode(generatedIds, skip_special_tokens = True)[0]  # type: ignore


def parseDate(inputDate: str) -> dict[str, Optional[int]]:
    # Replace all periods with spaces
    inputDate = inputDate.replace(".", " ")

    # "DD MM YY" -> "DD MM YYYY" [old Danish passorts]
    if len(inputDate.split(" ", 3)[2]) == 2:
        day, month, year = inputDate.split(" ", 3)

        yearPrefix = "19" if int(year) > int(str(datetime.now().year)[2:]) else "20"
        return parseDate(" ".join([day, month, yearPrefix + year]))

    try:
        dateTime = parser.parse(inputDate)
        return {
            "year": dateTime.year,
            "month": dateTime.month,
            "day": dateTime.day
        }
    except ValueError as e:
        # date time strings with alternate month spelling (assuming the alternate spelling is on the left of the "\") [Dutch IDs]
        if "/" in inputDate:
            prefix, rest = inputDate.split(" ", 1)
            return parseDate(prefix + " " + rest.split("/", 1)[1])

        return {
            "year": None,
            "month": None,
            "day": None
        }


def performOCR(images: list[Image.Image], classes: list[str]) -> dict[str, str]:
    detections: dict[str, str] = {}
    for i, image in enumerate(images):
        trOcrOutput = trOCR(image)

        if classes[i] == "date_of_birth":
            detections["date_of_birth_raw"] = trOcrOutput
            trOcrOutput = parseDate(trOcrOutput)

        detections[classes[i]] = trOcrOutput

    return detections
