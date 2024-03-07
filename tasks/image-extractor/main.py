from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future

import logging
import os
import json

from coretex import currentTaskRun, TaskRun, ComputerVisionDataset, ComputerVisionSample, CoretexImageAnnotation, createDataset

from src import sample_generator


def uploadSample(path: Path, datasetId: int) -> None:
    imagePath = path / "image.png"
    if not imagePath.exists():
        raise RuntimeError("Image not found")

    generatedSample = ComputerVisionSample.createComputerVisionSample(datasetId, imagePath)
    if generatedSample is None:
        logging.error(f">> [SyntheticDocumentGenerator] Failed to create sample from \"{imagePath}\"")
        return

    annotationPath = path / "annotation.json"
    if annotationPath.exists():
        with annotationPath.open("r") as file:
            annotation = CoretexImageAnnotation.decode(json.load(file))

        if not generatedSample.saveAnnotation(annotation):
            logging.error(f">> [SyntheticDocumentGenerator] Failed to save annotation for generated sample \"{generatedSample.name}\"")

    logging.info(f">> [SyntheticDocumentGenerator] Generated sample \"{generatedSample.name}\"")


def didGenerateSample(datasetId: int, future: Future[list[Path]]) -> None:
    try:
        for samplePath in future.result():
            uploadSample(samplePath, datasetId)
    except BaseException as exception:
        logging.error(f">> [SyntheticDocumentGenerator] Failed to generate sample. Reason: {exception}")
        logging.debug(exception, exc_info = exception)


def main() -> None:
    taskRun: TaskRun[ComputerVisionDataset] = currentTaskRun()
    taskRun.dataset.download()

    excludedClasses = taskRun.parameters.get("excludedClasses", [])
    if excludedClasses is not None:
        taskRun.dataset.classes.exclude(excludedClasses)

    parentClass = taskRun.dataset.classByName(taskRun.parameters["parentClass"])
    outputDatasetName = f"{taskRun.id} - {taskRun.dataset.name}"

    with createDataset(ComputerVisionDataset, outputDatasetName, taskRun.projectId) as outputDataset:
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
                uploader.submit(didGenerateSample, outputDataset.id, future)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
