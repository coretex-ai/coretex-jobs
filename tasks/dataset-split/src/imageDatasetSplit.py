import logging

from coretex import currentTaskRun, NetworkDataset, ImageDataset, ImageSample

from .utils import splitOriginalSamples


def imageDatasetSplit(originalDataset: ImageDataset, datasetParts: int, projectId: int) -> list[NetworkDataset]:
    splitSamples: list[list[ImageSample]] = splitOriginalSamples(originalDataset.samples, datasetParts)
    
    splitDatasets: list[NetworkDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = ImageDataset.createDataset(f"{currentTaskRun().id}-split-dataset-{index}", projectId)
        splitDataset.saveClasses(originalDataset.classes)

        for sample in sampleChunk:
            sample.unzip()
            addedSample = splitDataset.add(sample.imagePath)
            logging.info(f">> [Dataset Split] The sample \"{sample.name}\" has been added to the dataset \"{splitDataset.name}\"")

            tmpAnotation = sample.load().annotation
            if tmpAnotation is not None:
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

        logging.info(f">> [Dataset Split] New dataset named \"{splitDataset.name}\" contains {len(sampleChunk)} samples")

    return splitDatasets
