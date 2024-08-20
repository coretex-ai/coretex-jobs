from typing import Any, Optional
from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future

import random
import logging

from coretex import currentTaskRun, TaskRun, ImageDataset, ImageSample, CoretexImageAnnotation, createDataset

from src import sample_generator


def getRandomSamples(dataset: ImageDataset, count: int) -> list[ImageSample]:
    indexes: set[int] = set()

    while len(indexes) != count:
        indexes.add(random.randint(0, dataset.count - 1))

    return [dataset.samples[i] for i in indexes]


def didGenerateSample(
    dataset: ImageDataset,
    future: Future[tuple[Path, CoretexImageAnnotation, Optional[dict[str, Any]]]]
) -> None:

    try:
        imagePath, annotation, metadata = future.result()
        generatedSample = dataset.add(imagePath)

        if not generatedSample.saveAnnotation(annotation):
            logging.error(f">> [SyntheticImageGenerator] Failed to save annotation for generated sample \"{generatedSample.name}\"")
        else:
            logging.info(f">> [SyntheticImageGenerator] Generated sample \"{generatedSample.name}\"")

        if metadata is not None:
            generatedSample.saveMetadata(metadata)
    except BaseException as exception:
        logging.error(f">> [SyntheticImageGenerator] Failed to generate sample. Reason: {exception}")
        logging.debug(exception, exc_info = exception)


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()
    taskRun.dataset.download()

    backgroundDataset: ImageDataset = taskRun.parameters["backgroundDataset"]
    backgroundDataset.download()

    augmentationsPerImage = taskRun.parameters["augmentationsPerImage"]
    maxRotationAngle = taskRun.parameters["maxRotationAngle"]

    if augmentationsPerImage > backgroundDataset.count:
        logging.warning(
            ">> [SyntheticImageGenerator] \"augmentationsPerImage\" value: "
            f"{augmentationsPerImage} is higher than \"backgroundDataset\" "
            f"count: {backgroundDataset.count}. Limiting value to: {backgroundDataset.count}"
        )
        augmentationsPerImage = backgroundDataset.count

    # Seed used for multiple operations - needed for reproducibility
    random.seed(taskRun.parameters["seed"])

    with ExitStack() as stack:
        outputDatasetName = f"{taskRun.id}-{taskRun.dataset.name}"
        outputDataset = stack.enter_context(createDataset(ImageDataset, outputDatasetName, taskRun.projectId))
        outputDataset.saveClasses(taskRun.dataset.classes)

        executor = ProcessPoolExecutor(max_workers = 1)
        stack.enter_context(executor)

        uploader = ThreadPoolExecutor(max_workers = 4)
        stack.enter_context(uploader)

        for sample in taskRun.dataset.samples:
            for backgroundSample in getRandomSamples(backgroundDataset, augmentationsPerImage):
                backgroundSample.unzip()

                if maxRotationAngle > 0:
                    rotationAngle = random.randint(0, maxRotationAngle)
                else:
                    rotationAngle = 0

                future = executor.submit(sample_generator.generateSample,
                    sample,
                    backgroundSample.imagePath,
                    taskRun.parameters["minImageSize"],
                    taskRun.parameters["maxImageSize"],
                    rotationAngle
                )

                uploader.submit(didGenerateSample, outputDataset, future)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
