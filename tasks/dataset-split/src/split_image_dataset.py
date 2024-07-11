import logging
from coretex import ImageDataset, ImageSample, ImageDatasetClasses

from .utils import splitOriginalSamples


def splitImageDataset(originalDataset: ImageDataset, datasetParts: int, taskRunId: int, projectId: int) -> list[ImageDataset]:
    splitSamples: list[list[ImageSample]] = splitOriginalSamples(originalDataset.samples, datasetParts)

    splitDatasets: list[ImageDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        dependencies = [str(originalDataset.id), str(datasetParts), str(projectId), str(index)]
        try:
            splitDataset = ImageDataset.fetchCachedDataset(dependencies)
            logging.info(f">> [Dataset Split] The dataset with the name \"{splitDataset.name}\" has been fetched ")
        except ValueError:
            splitDataset = ImageDataset.createCacheDataset("split-dataset", dependencies, projectId)

            splitDatasetClasses = ImageDatasetClasses()

            for sample in sampleChunk:
                sample.unzip()

                addedSample = splitDataset.add(sample.imagePath)
                logging.info(f">> [Dataset Split] The sample \"{sample.name}\" has been added to the dataset \"{splitDataset.name}\"")

                tmpAnotation = sample.load().annotation
                if tmpAnotation is not None:
                    for instance in tmpAnotation.instances:
                        instanceClass = originalDataset.classes.classById(instance.classId)
                        if instanceClass is not None and instanceClass not in splitDatasetClasses:
                            splitDatasetClasses.extend([instanceClass])

                    splitDataset.saveClasses(splitDatasetClasses)

                    addedSample.saveAnnotation(tmpAnotation)
                    logging.info(f">> [Dataset Split] The anotation for sample \"{sample.name}\" has been added")

                try:
                    tmpMetadata = sample.loadMetadata()
                    addedSample.saveMetadata(tmpMetadata)
                    logging.info(f">> [Dataset Split] The metadata for sample \"{sample.name}\" has been added")
                except FileNotFoundError:
                    logging.info(f">> [Dataset Split] The metadata for sample \"{sample.name}\" was not found")
                except ValueError:
                    logging.info(f">> [Dataset Split] Invalid metadata type for sample \"{sample.name}\"")

        splitDatasets.append(splitDataset)

        logging.info(f">> [Dataset Split] New dataset named \"{splitDataset.name}\" contains {splitDataset.count} samples")

    return splitDatasets
