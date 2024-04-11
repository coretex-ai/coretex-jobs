import logging

from coretex import ImageDataset, ImageSample, ImageDatasetClasses,\
    currentTaskRun, TaskRun, createDataset


def isSampleAnnotated(sample: ImageSample, classes: ImageDatasetClasses) -> bool:
    sample.unzip()

    annotation = sample.load().annotation

    if annotation is None:
        return False

    for instance in annotation.instances:
        # If the sample has annotation for one of the classes from dataset it is annotated
        if classes.classById(instance.classId) is not None:
            return True

    return False


def copySamples(
    dataset: ImageDataset,
    samples: list[ImageSample]
) -> ImageDataset:

    for index, sample in enumerate(samples):
        logging.info(f">> [Coretex] Copying sample \"{sample.name}\" - {index + 1}/{len(samples)}")
        sample.unzip()

        copy = dataset.add(sample.imagePath)

        annotation = sample.load().annotation
        if annotation is not None:
            if not copy.saveAnnotation(annotation):
                logging.error("\tFailed to copy sample annotation, deleting...")

                if not copy.delete():
                    logging.error("\tFailed to delete sample")

    return dataset


def main() -> None:
    taskRun: TaskRun[ImageDataset] = currentTaskRun()

    dataset = taskRun.dataset
    dataset.download()

    # Create a dataset with only annotated samples
    annotatedSamples = [sample for sample in dataset.samples if isSampleAnnotated(sample, dataset.classes)]
    if len(annotatedSamples) == 0:
        raise RuntimeError(f"Provided dataset \"{dataset.name}\" has no annotated Samples")

    with createDataset(ImageDataset, f"{taskRun.id}-{dataset.name}", taskRun.projectId) as annotatedDataset:
        if not annotatedDataset.saveClasses(dataset.classes):
            raise RuntimeError("Failed to copy classes")

        # Copy annotated samples to annotated dataset
        copySamples(annotatedDataset, annotatedSamples)

        # Submit the created dataset as output
        taskRun.submitOutput("annotatedDataset", annotatedDataset)


if __name__ == "__main__":
    main()
