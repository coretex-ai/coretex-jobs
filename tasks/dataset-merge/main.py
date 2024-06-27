import logging
import re

from coretex import currentTaskRun, NetworkDataset, CustomDataset, ImageDataset, ImageDatasetClass, ImageDatasetClasses


def validateAndCorectEntityName(name: str) -> str:
    pattern = r"^[a-z0-9-]{3,50}$"
    logging.warning(f"pre {name}")
    if re.fullmatch(pattern, name) is not None:
        return name
    
    name = name.lower()
    name = name.replace("_", "-")
    name = re.sub(r"[^a-z0-9-]", "", name)

    if len(name) < 3:
        name = name.ljust(3, "-")
    if len(name) > 50:
        name = name[:50]
    logging.warning(f"posle {name}")
    return name


def customDatasetMerge(datasets: list[CustomDataset], projectId: int) -> NetworkDataset:
    mergeDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-merge-custom-dataset", projectId)
    
    for dataset in datasets:
        dataset.download()
        samples = dataset.samples
        
        for sample in samples:
            newName = validateAndCorectEntityName(sample.name)
            logging.warning(f"novo ime je {newName}")
            logging.warning(f"zipPath je {sample.zipPath.name}")
            addedSample = mergeDataset.add(sample.zipPath, newName[:newName.rfind("-")])

            logging.warning(f"ime dodatog sempla je {addedSample.name}")
            logging.warning(f"ime starog sempla je {sample.name}")
            logging.info(f">> [Dataset Merge] The sample \"{addedSample.name}\" has been added to the dataset \"{mergeDataset.name}\"")

    logging.info(f">> [Dataset Merge] New dataset named \"{mergeDataset.name}\" contains {mergeDataset.count} samples")
    
    return mergeDataset


def imageDatasetMerge(datasets: list[ImageDataset], projectId: int) -> NetworkDataset:
    mergeDataset = ImageDataset.createDataset(f"{currentTaskRun().id}-merge-image-dataset", projectId)

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
            newName = validateAndCorectEntityName(sample.name)
            logging.warning(f"novo ime je {newName}")
            logging.warning(f"imagePath je {sample.imagePath.name}")
            addedSample = mergeDataset.add(sample.imagePath, newName[:newName.rfind("-")])
            
            logging.warning(f"ime dodatog sempla je {addedSample.name}")
            logging.warning(f"ime starog sempla je {sample.name}")
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
    datasets = taskRun.parameters["datasetsList"]

    if isinstance(datasets[0], CustomDataset):
        logging.info(">> [Dataset Merge] Merging CustomDatasets...")
        mergeDataset = customDatasetMerge(datasets, projectId)
    
    if isinstance(datasets[0], ImageDataset):
        logging.info(">> [Dataset Merge] Merging ImageDatasets...")
        mergeDataset = imageDatasetMerge(datasets, projectId)

    taskRun.submitOutput("mergeDataset", mergeDataset)

if __name__ == "__main__":
    main()
