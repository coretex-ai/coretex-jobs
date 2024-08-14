from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future

import logging
import json
import os

from coretex import currentTaskRun, TaskRun, ImageDataset, CoretexImageAnnotation, createDataset

from src import sample_generator


def uploadSample(path: Path, dataset: ImageDataset) -> None:
    imagePath = path / "image.png"
    if not imagePath.exists():
        raise RuntimeError("Image not found")

    try:
        generatedSample = dataset.add(imagePath)
    except BaseException as ex:
        logging.error(f">> [ImageExtractor] Failed to create sample from \"{imagePath}\" - \"{ex}\"")
        return

    annotationPath = path / "annotation.json"
    if annotationPath.exists():
        with annotationPath.open("r") as file:
            annotation = CoretexImageAnnotation.decode(json.load(file))

        if not generatedSample.saveAnnotation(annotation):
            logging.error(f">> [ImageExtractor] Failed to save annotation for generated sample \"{generatedSample.name}\"")

    metadataPath = path / "metadata.json"
    if metadataPath.exists():
        with metadataPath.open("r") as file:
            metadata = json.load(file)

        try:
            generatedSample.saveMetadata(metadata)
        except ValueError:
            logging.info(f">> [ImageExtractor] Invalid metadata type for sample \"{generatedSample.name}\"")

    logging.info(f">> [ImageExtractor] Generated sample \"{generatedSample.name}\"")


def didGenerateSample(dataset: ImageDataset, future: Future[list[Path]]) -> None:
    try:
        for samplePath in future.result():
            uploadSample(samplePath, dataset)
    except BaseException as exception:
        logging.error(f">> [ImageExtractor] Failed to generate sample. Reason: {exception}")
        logging.debug(exception, exc_info = exception)


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()
    taskRun.dataset.download()

    excludedClasses = taskRun.parameters.get("excludedClasses", [])
    if excludedClasses is not None:
        taskRun.dataset.classes.exclude(excludedClasses)

    parentClass = taskRun.dataset.classByName(taskRun.parameters["parentClass"])
    outputDatasetName = f"{taskRun.id}-{taskRun.dataset.name}"

    with createDataset(ImageDataset, outputDatasetName, taskRun.projectId) as outputDataset:
        outputDataset.saveClasses(taskRun.dataset.classes)

        with ExitStack() as stack:
            executor = ProcessPoolExecutor(max_workers = os.cpu_count())
            stack.enter_context(executor)

            uploader = ThreadPoolExecutor(max_workers = 4)
            stack.enter_context(uploader)

            for sample in taskRun.dataset.samples:
                # Process sample
                future = executor.submit(sample_generator.generateSample, sample, parentClass)

                # Upload sample
                uploader.submit(didGenerateSample, outputDataset, future)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
