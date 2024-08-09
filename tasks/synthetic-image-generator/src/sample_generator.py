from pathlib import Path

import random
import uuid

from PIL import Image, ImageOps
from coretex import ImageSample, CoretexImageAnnotation, CoretexSegmentationInstance, folder_manager
from coretex.utils import cropToWidth

# import matplotlib.pyplot as plt

from . import utils


def updateAnnotationPosition(
    annotation: CoretexImageAnnotation,
    imageWidth: int,
    imageHeight: int,
    parentAnnotationWidth: int,
    parentAnnotationHeight: int,
    parentAnnotationTopX: int,
    parentAnnotationTopY: int
) -> CoretexImageAnnotation:

    resizedInstances: list[CoretexSegmentationInstance] = []

    for instance in annotation.instances:
        resizedInstance = utils.resizeInstance(instance, annotation.width, annotation.height, parentAnnotationWidth, parentAnnotationHeight)
        utils.offsetSegmentations(resizedInstance, parentAnnotationTopX, parentAnnotationTopY)

        resizedInstances.append(resizedInstance)

    return CoretexImageAnnotation.create(annotation.name, imageWidth, imageHeight, resizedInstances)


def generateSample(
    sample: ImageSample,
    backgroundImagePath: Path,
    minImageSize: float,
    maxImageSize: float,
    rotationAngle: int
) -> tuple[Path, CoretexImageAnnotation]:

    sample.unzip()

    data = sample.load()

    annotation = data.annotation
    if annotation is None:
        raise RuntimeError(f"Sample \"{sample.name}\" has no annotations")

    image = Image.fromarray(data.image)
    backgroundImage = ImageOps.exif_transpose(Image.open(backgroundImagePath))
    if backgroundImage is None:
        raise ValueError(f"Failed to open background image. ID: {backgroundImagePath.parent.name}")

    # Resize image
    parentAnnotationWidth = int(backgroundImage.width * random.uniform(minImageSize, maxImageSize))
    parentAnnotationHeight = int(parentAnnotationWidth * (image.height / image.width))
    parentAnnotationTopX = random.randint(0, backgroundImage.width - parentAnnotationWidth)
    parentAnnotationTopY = random.randint(0, backgroundImage.height - parentAnnotationHeight)
    parentAnnotationCentroid = parentAnnotationTopX + parentAnnotationWidth // 2, parentAnnotationTopY + parentAnnotationHeight // 2

    image = image.resize((parentAnnotationWidth, parentAnnotationHeight), Image.Resampling.LANCZOS)

    # Paste image to background image
    rotatedParentAnnotationMask = Image.new("L", image.size, 255).rotate(rotationAngle, expand = True)
    image = image.rotate(rotationAngle, expand = True)

    backgroundImage.paste(image, (parentAnnotationTopX, parentAnnotationTopY), rotatedParentAnnotationMask)

    updatedAnnotation = updateAnnotationPosition(
        annotation,
        backgroundImage.width,
        backgroundImage.height,
        parentAnnotationWidth,
        parentAnnotationHeight,
        parentAnnotationTopX,
        parentAnnotationTopY
    )

    for instance in updatedAnnotation.instances:
        instance.rotateSegmentations(rotationAngle, parentAnnotationCentroid)

    # expand = True for PIL during rotation so image gets resized
    # calculate diff and offset annotations by that amount
    widthDiff = image.width - parentAnnotationWidth
    heightDiff = image.height - parentAnnotationHeight

    for instance in updatedAnnotation.instances:
        utils.offsetSegmentations(instance, widthDiff // 2, heightDiff // 2)

    # For debugging only
    # plt.imshow(backgroundImage)
    # for instance in updatedAnnotation.instances:
    #     for segmentation in instance.segmentations:
    #         plt.plot(
    #             [p for i, p in enumerate(segmentation) if i % 2 == 0],
    #             [p for i, p in enumerate(segmentation) if i % 2 != 0]
    #         )
    # plt.show()

    identifier = uuid.uuid4()
    imagePath = folder_manager.temp / f"{identifier}.jpeg"
    backgroundImage.save(imagePath)

    return imagePath, updatedAnnotation
