from typing import Optional
from dateutil import parser
from datetime import datetime

import logging

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


modelVersion = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(modelVersion)
model = VisionEncoderDecoderModel.from_pretrained(modelVersion)


def trOCR(image: Image.Image) -> str:
    pixelValues = processor(image.convert("RGB"), return_tensors = "pt").pixel_values
    generatedIds = model.generate(pixelValues)
    return processor.batch_decode(generatedIds, skip_special_tokens = True)[0]  # type: ignore


def parseDate(inputDate: str) -> tuple[str, dict[str, Optional[int]]]:
    # Replace all periods with spaces
    inputDate = inputDate.replace(".", " ")

    try:
        day, month, year = inputDate.split(" ", 3)
    except ValueError as e:
        logging.debug(f">> [DocumentOCR] \"{inputDate}\" could not be separated into dat/month/year, returning None for DOB fields. Error: {e}")
        return (
            inputDate,
            {
                "year": None,
                "month": None,
                "day": None
            }
        )

    # If month has two spellings (e.g. MAA/MAR), but the slash was not detected, add slash in middle
    if len(month) == 6:
        month = month[:3] + "/" + month[3:]
        inputDate = " ".join([day, month, year])

    # "DD MM YY" -> "DD MM YYYY" [old Danish passorts]
    if len(year) == 2:
        yearPrefix = "19" if int(year) > int(str(datetime.now().year)[2:]) else "20"
        return parseDate(" ".join([day, month, yearPrefix + year]))

    try:
        dateTime = parser.parse(inputDate)
        return (
            inputDate,
            {
                "year": dateTime.year,
                "month": dateTime.month,
                "day": dateTime.day
            }
        )
    except ValueError as e:
        logging.debug(f">> [DocumentOCR] \"{inputDate}\" could not be parsed as date-time, performing additional preprocessing. Error: {e}", exc_info = e)
        # date time strings with alternate month spelling (assuming the alternate spelling is on the left of the "\") [Dutch IDs]
        if "/" in inputDate:
            prefix, rest = inputDate.split(" ", 1)
            return parseDate(prefix + " " + rest.split("/", 1)[1])

        return (
            inputDate,
            {
                "year": None,
                "month": None,
                "day": None
            }
        )


def performOCR(images: list[Image.Image], classes: list[str]) -> dict[str, str]:
    detections: dict[str, str] = {}
    for i, image in enumerate(images):
        trOcrOutput = trOCR(image)

        if classes[i] == "date_of_birth":
            dateOfBirthRaw, trOcrOutput = parseDate(trOcrOutput)
            detections["date_of_birth_raw"] = dateOfBirthRaw

        detections[classes[i]] = trOcrOutput

    return detections


print(parseDate("xd xd dddd"))
