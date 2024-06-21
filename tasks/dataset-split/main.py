import csv
import logging
import zipfile

from coretex import currentTaskRun, folder_manager, ImageDataset, CustomDataset, SequenceDataset, ImageSample, CustomSample, SequenceSample, NetworkDataset
from coretex.networking import NetworkRequestError

from utils import splitOriginalSamples


def imageDatasetSplit(originalDataset: ImageDataset, datasetParts: int, projectID: int) -> list[NetworkDataset]:
    splitSamples: list[list[ImageSample]] = splitOriginalSamples(originalDataset, datasetParts)
    
    splitDatasets: list[NetworkDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = ImageDataset.createDataset(f"{currentTaskRun().id}-splitDataset-{index}", projectID)
        splitDataset.saveClasses(originalDataset.classes)

        for sample in sampleChunk:
            sample.unzip()
            addedSample = splitDataset.add(sample.imagePath)
            logging.info(f' >> [Dataset Split] The sample "{sample.name}" has been added to the dataset "{splitDataset.name}"')

            tmpAnotation = sample.load().annotation
            if tmpAnotation is not None:
                addedSample.saveAnnotation(tmpAnotation)
                logging.info(f' >> [Dataset Split] The anotation for sample "{sample.name}" has been added')

            try:
                tmpMetadata = sample.loadMetadata()
                addedSample.saveMetadata(tmpMetadata)
                logging.info(f' >> [Dataset Split] The metadata for sample "{sample.name}" has been added')
            except FileNotFoundError:
                logging.info(f' >> [Dataset Split] The metadata for sample "{sample.name}" was not found')
            except ValueError:
                logging.info(f' >> [Dataset Split] Invalid metadata type for sample "{sample.name}"')
        
        splitDatasets.append(splitDataset)

        logging.info(f' >> [Dataset Split] New dataset named "{splitDataset.name}" contains {len(sampleChunk)} samples')

    return splitDatasets


def customDatasetSplit(originalDataset: SequenceDataset, datasetParts: int, projectID: int) -> list[NetworkDataset]:  
    splitSamples: list[list[CustomSample]] = splitOriginalSamples(originalDataset, datasetParts)
    
    splitDatasets: list[NetworkDataset] = []

    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-splitDataset-{index}", projectID)

        for sample in sampleChunk:
            sample.unzip()
            splitDataset.add(sample.zipPath)
            logging.info(f' >> [Dataset Split] The sample "{sample.name}" has been added to the dataset "{splitDataset.name}"')

        splitDatasets.append(splitDataset)

        logging.info(f' >> [Dataset Split] New dataset named "{splitDataset.name}" contains {len(sampleChunk)} samples')

    return splitDatasets


def sequenceDatasetSplit(originalDataset: SequenceDataset, datasetParts: int, projectID: int) -> list[NetworkDataset]:
    splitSamples: list[list[SequenceSample]] = splitOriginalSamples(originalDataset, datasetParts)
    
    metadataAddress = list(originalDataset.metadata.load().folderContent)[0] #address where the file metadata is located in the form of a string
  
    with open(metadataAddress, mode = "r", newline = "") as file:
        reader = csv.DictReader(file)
        originalMetadata: list[dict] = []
        
        for row in reader:
            originalMetadata.append(dict(row))

    splitDatasets: list[NetworkDataset] = []
    
    for index, sampleChunk in enumerate(splitSamples):
        splitDataset = CustomDataset.createDataset(f"{currentTaskRun().id}-splitDataset-{index}", projectID)
        splitMetadataList: list[dict] = []
        
        for sample in sampleChunk:
            sample.unzip()
            splitDataset.add(sample.zipPath)
            logging.info(f' >> [Dataset Split] The sample "{sample.name}" has been added to the dataset "{splitDataset.name}"')
            
            fieldNames = list(originalMetadata[0].keys())
            
            for oneMetadata in originalMetadata:
                if sample.name.startswith(oneMetadata[fieldNames[0]].split("_")[0]):
                    splitMetadataList.append(oneMetadata)

        metadataNameCSV = f"_metadata_{index}.csv"
        metadataCSV = folder_manager.temp / metadataNameCSV
        with open(metadataCSV, "w", newline = "") as file:
            writer = csv.DictWriter(file, fieldnames=fieldNames)
            writer.writeheader()
            writer.writerows(splitMetadataList)

        metadataNameZIP = f"_metadata_{index}.zip"
        metadataZIP = folder_manager.temp / metadataNameZIP
        with zipfile.ZipFile(metadataZIP, "w") as zipFile:
            zipFile.write(metadataCSV)

        splitDataset.add(metadataZIP)
        logging.info(f' >> [Dataset Split] The _metadata sample "{metadataNameZIP}" has been added to the dataset "{splitDataset.name}"')

        splitDatasets.append(splitDataset)
        logging.info(f' >> [Dataset Split] New dataset named "{splitDataset.name}" contains {len(sampleChunk)} samples')
    
    return splitDatasets


def main() -> None:
    taskRun = currentTaskRun()
    originalDataset = taskRun.dataset
    datasetParts = taskRun.parameters["datasetParts"]
    projectID = taskRun.projectId

    n = originalDataset.count
    
    if n <= datasetParts:
        raise ValueError("Number of samples is smaller than the number you want to divide the dataset")
    if datasetParts < 2:
        raise ValueError("Dataset can be divided into at least two parts")

    splitDatasets: list[NetworkDataset]

    if isinstance(originalDataset, ImageDataset):
        logging.info(f" >> [Dataset Split] Divisioning ImageDataset {originalDataset.name}...")
        splitDatasets = imageDatasetSplit(originalDataset, datasetParts, projectID)

    if isinstance(originalDataset, CustomDataset):
        try:
            taskRun.setDatasetType(SequenceDataset)
            originalDataset = taskRun.dataset
            logging.info(f" >> [Dataset Split] Divisioning SequenceDataset {originalDataset.name}...")
            splitDatasets = sequenceDatasetSplit(originalDataset, datasetParts, projectID)
        except:
            logging.info(f" >> [Dataset Split] Divisioning CustomDataset {originalDataset.name}...")
            splitDatasets = customDatasetSplit(originalDataset, datasetParts, projectID) 

    splitDatasetIDs = [ds.id for ds in splitDatasets]

    try:
        taskRun.submitOutput("splitDatasetIDs", splitDatasetIDs)
    except NetworkRequestError as e:
        logging.warning(f" >> [Dataset Split] Error while submitting the output value: {e}")
        

if __name__ == "__main__":
    main()
