from typing import Optional
from pathlib import Path

import uuid
import json

from coretex import ImageSample, folder_manager, ImageDatasetClass
from PIL import Image

from . import image_extractor


def generateSample(sample: ImageSample, parentClass: Optional[ImageDatasetClass]) -> list[Path]:
    sample.unzip()

    data = sample.load()
    image = data.image
    annotation = data.annotation

    if annotation is None:
        raise RuntimeError(f"Sample \"{sample.name}\" has no annotations")

    try:
        imagePaths: list[Path] = []

        for transformedImage, transformedAnnotation in image_extractor.extractImages(image, annotation, parentClass):
            # Create a directory where sample will be stored
            identifier = uuid.uuid4()
            samplePath = folder_manager.createTempFolder(str(identifier))

            # Save image
            imagePath = samplePath / "image.png"
            Image.fromarray(transformedImage).save(imagePath)

            # Save annotation - annotation is optional
            if transformedAnnotation is not None:
                annotationPath = samplePath / "annotation.json"
                with annotationPath.open("w") as file:
                    json.dump(transformedAnnotation.encode(), file)

            imagePaths.append(samplePath)

        return imagePaths
    except ValueError as e:
        raise ValueError(f"Could not generate image for sample \"{sample.name}\". Reason: {e}")
