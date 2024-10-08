import logging

from coretex import currentTaskRun, ProjectType, NetworkDataset, CustomDataset, ImageDataset, ImageDatasetClasses


ALLOWED_PROJECT_TYPES = [ProjectType.computerVision, ProjectType.other]


def mergeCustomDataset(datasets: list[CustomDataset], taskRunId: int, projectId: int) -> CustomDataset:
    mergedDataset = CustomDataset.createDataset(f"{taskRunId}-merge-custom-dataset", projectId)

    for dataset in datasets:
        dataset.download()
        samples = dataset.samples

        for sample in samples:
            addedSample = mergedDataset.add(sample.zipPath)
            logging.info(f">> [Dataset Merge] The sample \"{addedSample.name}\" has been added to the dataset \"{mergedDataset.name}\"")

    logging.info(f">> [Dataset Merge] New dataset named \"{mergedDataset.name}\" contains {mergedDataset.count} samples")

    return mergedDataset


def mergeImageDataset(datasets: list[ImageDataset], taskRunId: int, projectId: int) -> ImageDataset:
    mergedDataset = ImageDataset.createDataset(f"{taskRunId}-merge-image-dataset", projectId)

    allClasses = ImageDatasetClasses()

    for dataset in datasets:
        for oneClass in dataset.classes:
            originalClass = allClasses.classByLabel(oneClass.label)
            if originalClass is not None:
                originalClass.classIds.extend(oneClass.classIds)
                originalClass.classIds = list(set(originalClass.classIds))
            else:
                allClasses.append(oneClass)
                logging.info(f">> [Dataset Merge] The class \"{oneClass.label}\" has been saved to the new dataset \"{mergedDataset.name}\"")

    mergedDataset.saveClasses(allClasses)

    for dataset in datasets:
        dataset.download()
        samples = dataset.samples

        for sample in samples:
            sample.unzip()
            addedSample = mergedDataset.add(sample.imagePath)
            logging.info(f">> [Dataset Merge] The sample \"{addedSample.name}\" has been added to the dataset \"{mergedDataset.name}\"")

            tmpAnotation = sample.load().annotation
            if tmpAnotation is not None:
                addedSample.saveAnnotation(tmpAnotation)
                logging.info(f">> [Dataset Merge] The anotation for sample \"{addedSample.name}\" has been added")

            try:
                tmpMetadata = sample.loadMetadata()
                addedSample.saveMetadata(tmpMetadata)
                logging.info(f">> [Dataset Merge] The metadata for sample \"{addedSample.name}\" has been added")
            except FileNotFoundError:
                logging.info(f">> [Dataset Merge] The metadata for sample \"{addedSample.name}\" was not found")
            except ValueError:
                logging.info(f">> [Dataset Merge] Invalid metadata type for sample \"{addedSample.name}\"")

    logging.info(f">> [Dataset Merge] New dataset named \"{mergedDataset.name}\" contains {mergedDataset.count} samples")

    return mergedDataset


def main() -> None:
    taskRun = currentTaskRun()
    projectId = taskRun.projectId
    taskRunId = taskRun.id
    datasets = taskRun.parameters["datasets"]

    if len(datasets) < 2:
        raise RuntimeError("The number of datasets to merge must be at least two")

    if taskRun.projectType not in ALLOWED_PROJECT_TYPES:
        raise RuntimeError(f"Currently, merging datasets is allowed for projects of the following types: {ALLOWED_PROJECT_TYPES}.\nYour project is of type: {taskRun.projectType}")

    mergedDataset: NetworkDataset

    if taskRun.projectType == ProjectType.computerVision:
        if False in [hasattr(dataset, "classes") for dataset in datasets]:
            raise TypeError("The datasets you provided for merging are not of the ImageDataset type")

        logging.info(">> [Dataset Merge] Merging ImageDatasets...")
        mergedDataset = mergeImageDataset(datasets, taskRunId, projectId)
    elif taskRun.projectType == ProjectType.other:
        logging.info(">> [Dataset Merge] Merging CustomDatasets...")
        mergedDataset = mergeCustomDataset(datasets, taskRunId, projectId)

    taskRun.submitOutput("mergedDataset", mergedDataset)


if __name__ == "__main__":
    main()
