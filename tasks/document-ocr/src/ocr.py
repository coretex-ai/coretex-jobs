from pathlib import Path

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from coretex import TaskRun

from .utils import createArtifact


# TrOCR
modelVersion = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(modelVersion)
model = VisionEncoderDecoderModel.from_pretrained(modelVersion)


def trOCR(image: Image.Image) -> str:
    pixelValues = processor(image.convert("RGB"), return_tensors = "pt").pixel_values
    generatedIds = model.generate(pixelValues)
    return processor.batch_decode(generatedIds, skip_special_tokens = True)[0]  # type: ignore


def performOCR(images: list[Image.Image], classes: list[str], outputDir: Path, taskRun: TaskRun) -> None:
    for i, image in enumerate(images):
        classResultPath = outputDir / classes[i]
        classResultPath.mkdir(exist_ok = True)

        trOcrOutputPath = classResultPath.joinpath("ocr.txt")
        with trOcrOutputPath.open("w") as file:
            file.write(trOCR(image))

        createArtifact(taskRun, trOcrOutputPath, str(trOcrOutputPath.relative_to(outputDir.parent)))
