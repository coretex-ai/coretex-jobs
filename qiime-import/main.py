from coretex import Experiment, CustomDataset, SequenceDataset, folder_manager
from coretex.project import initializeProject

from src.multiplexed import importMultiplexed
from src.demultiplexed import importDemultiplexed


def main(experiment: Experiment[CustomDataset]):
    dataset = experiment.dataset
    dataset.download()

    outputDir = folder_manager.createTempFolder("import_output")

    if experiment.parameters["barcodeColumn"]:
        importMultiplexed(dataset, experiment, outputDir)
    else:
        importDemultiplexed(SequenceDataset.fetchById(dataset.id), experiment, outputDir)


if __name__ == "__main__":
    initializeProject(main)
