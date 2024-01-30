from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future

import random
import logging
import os

from coretex import currentTaskRun, TaskRun, ComputerVisionDataset, ComputerVisionSample, CoretexImageAnnotation, createDataset

from src import sample_generator


def getRandomSamples(dataset: ComputerVisionDataset, count: int) -> list[ComputerVisionSample]:
    indexes: set[int] = set()

    while len(indexes) != count:
        indexes.add(random.randint(0, dataset.count - 1))

    return [dataset.samples[i] for i in indexes]


def didGenerateSample(datasetId: int, future: Future[tuple[Path, CoretexImageAnnotation]]) -> None:
    try:
        imagePath, annotation = future.result()

        generatedSample = ComputerVisionSample.createComputerVisionSample(datasetId, imagePath)
        if generatedSample is not None:
            if not generatedSample.saveAnnotation(annotation):
                logging.error(f">> [SyntheticDocumentGenerator] Failed to save annotation for generated sample \"{generatedSample.name}\"")
            else:
                logging.info(f">> [SyntheticDocumentGenerator] Generated sample \"{generatedSample.name}\"")
        else:
            logging.error(f">> [SyntheticDocumentGenerator] Failed to create sample from \"{imagePath}\"")
    except BaseException as exception:
        logging.error(f">> [SyntheticDocumentGenerator] Failed to generate sample. Reason: {exception}")
        logging.debug(exception, exc_info = exception)


def main() -> None:
    taskRun: TaskRun[ComputerVisionDataset] = currentTaskRun()
    taskRun.dataset.download()

    backgroundDataset: ComputerVisionDataset = taskRun.parameters["backgroundDataset"]
    backgroundDataset.download()

    imagesPerDocument = taskRun.parameters["imagesPerDocument"]
    if imagesPerDocument > backgroundDataset.count:
        logging.warning(
            ">> [SyntheticDocumentGenerator] \"imagesPerDocument\" value: "
            f"{imagesPerDocument} is higher than \"backgroundDataset\" "
            f"count: {backgroundDataset.count}. Limiting value to: {backgroundDataset.count}"
        )
        imagesPerDocument = backgroundDataset.count

    # Seed used for multiple operations - needed for reproducibility
    random.seed(taskRun.parameters["seed"])

    outputDatasetName = f"{taskRun.id} - {taskRun.dataset.name}"
    with createDataset(ComputerVisionDataset, outputDatasetName, taskRun.projectId) as outputDataset:
        outputDataset.saveClasses(taskRun.dataset.classes)

        with ExitStack() as stack:
            executor = ProcessPoolExecutor(max_workers = os.cpu_count())
            stack.enter_context(executor)

            uploader = ThreadPoolExecutor(max_workers = 4)
            stack.enter_context(uploader)

            for sample in taskRun.dataset.samples:
                sample.unzip()

                for backgroundSample in getRandomSamples(backgroundDataset, taskRun.parameters["imagesPerDocument"]):
                    backgroundSample.unzip()

                    future = executor.submit(sample_generator.generateSample,
                        sample,
                        backgroundSample.imagePath,
                        taskRun.dataset.classes,
                        taskRun.parameters["minDocumentSize"],
                        taskRun.parameters["maxDocumentSize"]
                    )

                    uploader.submit(didGenerateSample, outputDataset.id, future)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
