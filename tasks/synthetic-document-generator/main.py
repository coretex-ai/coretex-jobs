from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future

import random
import logging
import os
import functools

from coretex import currentTaskRun, TaskRun, ComputerVisionDataset, ComputerVisionSample, CoretexImageAnnotation

from src import sample_generator


def getRandomSamples(dataset: ComputerVisionDataset, count: int) -> list[ComputerVisionSample]:
    indexes: set[int] = set()

    while len(indexes) != count:
        indexes.add(random.randint(0, dataset.count - 1))

    return [dataset.samples[i] for i in indexes]


def didGenerateSample(datasetId: int, future: Future[tuple[Path, CoretexImageAnnotation]]) -> None:
    exception = future.exception()
    if exception is not None:
        logging.error(f">> [SyntheticDocumentGenerator] Failed to generate sample. Reason: {exception}")
        logging.debug(exception, exc_info = exception)
        return

    imagePath, annotation = future.result()

    generatedSample = ComputerVisionSample.createComputerVisionSample(datasetId, imagePath)
    if generatedSample is not None:
        generatedSample.download()
        generatedSample.unzip()

        if not generatedSample.saveAnnotation(annotation):
            logging.error(f">> [SyntheticDocumentGenerator] Failed to save annotation for generated sample \"{generatedSample.name}\"")
        else:
            logging.info(f">> [SyntheticDocumentGenerator] Generated sample \"{generatedSample.name}\"")
    else:
        logging.error(f">> [SyntheticDocumentGenerator] Failed to create sample from \"{imagePath}\"")


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

    outputDataset = ComputerVisionDataset.createDataset(f"{taskRun.id} - {taskRun.dataset.name}", taskRun.projectId)
    if outputDataset is None:
        raise ValueError("Failed to create output dataset")

    outputDataset.saveClasses(taskRun.dataset.classes)

    with ProcessPoolExecutor(max_workers = os.cpu_count()) as executor:
        for sample in taskRun.dataset.samples:
            sample.unzip()

            for backgroundSample in getRandomSamples(backgroundDataset, imagesPerDocument):
                backgroundSample.unzip()
                backgroundData = backgroundSample.load()

                future = executor.submit(sample_generator.generateSample,
                    sample,
                    backgroundData.image,
                    taskRun.dataset.classes,
                    taskRun.parameters["minDocumentSize"],
                    taskRun.parameters["maxDocumentSize"]
                )

                future.add_done_callback(functools.partial(didGenerateSample, outputDataset.id))

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
