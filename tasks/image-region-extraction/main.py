import logging

from PIL import Image
from PIL.Image import Image as PILImage

import numpy as np
import cv2

from coretex import (
    ImageDataset, ImageSample, ImageDatasetClasses, CoretexSegmentationInstance,
    folder_manager, currentTaskRun, createDataset
)


class Point2D:

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class Rect:

    def __init__(self, tl: Point2D, tr: Point2D, br: Point2D, bl: Point2D) -> None:
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl

    @property
    def witdh(self) -> int:
        return self.tr.x - self.tl.x

    @property
    def height(self) -> int:
        return self.bl.y - self.tl.y

    def numpy(self) -> np.ndarray:
        return np.array([
            [self.tl.x, self.tl.y],
            [self.tr.x, self.tr.y],
            [self.br.x, self.br.y],
            [self.bl.x, self.bl.y]
        ], dtype = np.float32)

    def center(self) -> tuple[int, int]:
        temp = np.sum(self.numpy(), axis = 0) // 4
        return temp

    @classmethod
    def sortRectPoints(cls, points: list[Point2D]) -> list[Point2D]:
        xSorted = sorted(points, key = lambda point: point.x)

        left = xSorted[:2]
        left.sort(key = lambda point: point.y)
        tl, bl = left

        right = xSorted[2:]
        right.sort(key = lambda point: point.y)
        tr, br = right

        return [
            tl, tr, br, bl
        ]

    @classmethod
    def extractRectangle(cls, mask: np.ndarray) -> 'Rect':
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 1:
            raise ValueError("Found more than one contour")

        epsilon = 0.02 * cv2.arcLength(contours[0], True)
        rectangle = cv2.approxPolyDP(contours[0], epsilon, True)

        points: list[Point2D] = []
        for point in rectangle:
            points.append(Point2D(point[0][0], point[0][1]))

        points = cls.sortRectPoints(points)
        return cls(*points)


def generateExtractedImage(
    image: np.ndarray,
    segmentationMask: np.ndarray
) -> Image.Image:

    rgbaImage = Image.fromarray(image).convert("RGBA")

    segmentedImage = np.asarray(rgbaImage) * segmentationMask[..., None]  # reshape segmentationMask for broadcasting
    segmentedImage = Image.fromarray(segmentedImage)

    rectangle = Rect.extractRectangle(segmentationMask)

    width = rectangle.witdh
    height = rectangle.height

    transformed = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    transformMatrix = cv2.getPerspectiveTransform(rectangle.numpy(), transformed)
    segmentedImage = Image.fromarray(cv2.warpPerspective(np.array(segmentedImage), transformMatrix, (width, height)))

    alpha = segmentedImage.getchannel("A")
    bbox = alpha.getbbox()
    croppedImage = segmentedImage.crop(bbox)

    return croppedImage


def processSample(
    sample: ImageSample,
    classes: ImageDatasetClasses,
    excludedClasses: list[str],
    outputDataset: ImageDataset
) -> tuple[PILImage, list[CoretexSegmentationInstance]]:

    extractedImagesDir = folder_manager.createTempFolder(sample.name)
    sampleData = sample.load()

    annotation = sampleData.annotation
    if annotation is None:
        raise RuntimeError(f"CTX sample dataset sample id: {sample.id} image doesn't exist!")

    for i, instance in enumerate(annotation.instances):
        if excludedClasses is not None and classes.classById(instance.classId).label in excludedClasses:
            continue

        foregroundMask = instance.extractBinaryMask(sampleData.image.shape[1], sampleData.image.shape[0])
        extractedImage = generateExtractedImage(sampleData.image, foregroundMask, True)

        extractedImagePath = extractedImagesDir / f"{sample.name}-{classes.classById(instance.classId)}-{i}.png"
        extractedImage.save(extractedImagePath)
        ImageSample.createImageSample(outputDataset.id, extractedImagePath)


def main() -> None:
    taskRun = currentTaskRun()

    imagesDataset = taskRun.dataset
    imagesDataset: ImageDataset
    imagesDataset.download()

    with createDataset(ImageDataset, f"{taskRun.id}-ExtractedImages", taskRun.projectId) as outputDataset:
        outputDataset.saveClasses(imagesDataset.classes)

        for imageSample in imagesDataset.samples:
            imageSample.unzip()

            logging.info(f">> [RegionExtraction] Extracting annotated regions for {imageSample.name}")
            processSample(
                imageSample,
                imagesDataset.classes,
                taskRun.parameters["excludedClasses"],
                outputDataset
            )

        taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
