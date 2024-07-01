import logging

from coretex import currentTaskRun, ProjectType, NetworkDataset, CustomDataset, ImageDataset, ImageDatasetClasses


def customDatasetMerge(datasets: list[CustomDataset], taskRunId: int, projectId: int) -> NetworkDataset:
    mergeDataset = CustomDataset.createDataset(f"{taskRunId}-merge-custom-dataset", projectId)

    for dataset in datasets:
        dataset.download()
        samples = dataset.samples

        for sample in samples:
            addedSample = mergeDataset.add(sample.zipPath)
            logging.info(f">> [Dataset Merge] The sample \"{addedSample.name}\" has been added to the dataset \"{mergeDataset.name}\"")

    logging.info(f">> [Dataset Merge] New dataset named \"{mergeDataset.name}\" contains {mergeDataset.count} samples")

    return mergeDataset


def imageDatasetMerge(datasets: list[ImageDataset], taskRunId: int, projectId: int) -> NetworkDataset:
    mergeDataset = ImageDataset.createDataset(f"{taskRunId}-merge-image-dataset", projectId)

    allClasses = ImageDatasetClasses()

    for dataset in datasets:

        for oneClass in dataset.classes:
            if oneClass.label in allClasses.labels:
                originalClass = [cls for cls in allClasses if oneClass.label == cls.label][0]
                originalClass.classIds.extend(oneClass.classIds)
                originalClass.classIds = list(set(originalClass.classIds))
            else:
                allClasses.append(oneClass)
                logging.info(f">> [Dataset Merge] The class \"{oneClass.label}\" has been saved to the new dataset \"{mergeDataset.name}\"")

    mergeDataset.saveClasses(allClasses)

    for dataset in datasets:
        dataset.download()
        samples = dataset.samples

        for sample in samples:
            sample.unzip()
            addedSample = mergeDataset.add(sample.imagePath)
            logging.info(f">> [Dataset Merge] The sample \"{addedSample.name}\" has been added to the dataset \"{mergeDataset.name}\"")

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

    logging.info(f">> [Dataset Merge] New dataset named \"{mergeDataset.name}\" contains {mergeDataset.count} samples")

    return mergeDataset


def main() -> None:
    taskRun = currentTaskRun()
    projectId = taskRun.projectId
    taskRunId = taskRun.id
    datasets = taskRun.parameters["datasetsList"]

    if len(datasets) < 2:
        raise ValueError("The number of datasets to merge must be at least two")

    if taskRun.projectType == ProjectType.computerVision:
        if sum([hasattr(dataset, "classes") for dataset in datasets]) == len(datasets):
            logging.info(">> [Dataset Merge] Merging ImageDatasets...")
            mergeDataset = imageDatasetMerge(datasets, taskRunId, projectId)
        else:
            raise FileNotFoundError("The datasets you provided for merging are not of the ImageDataset type")

    elif taskRun.projectType == ProjectType.other:
        logging.info(">> [Dataset Merge] Merging CustomDatasets...")
        mergeDataset = customDatasetMerge(datasets, taskRunId, projectId)

    else:
        raise ValueError("Currently, merging datasets is allowed for projects of the following types: ComputerVision and Other")

    taskRun.submitOutput("mergeDataset", mergeDataset)

if __name__ == "__main__":
    main()
