from coretex import SequenceDataset, CustomDataset, Experiment, folder_manager
from coretex.project import initializeProject

from src.multiplexed import demultiplexing
from src.demultiplexed import importDemultiplexedSamples


def main(experiment: Experiment[SequenceDataset]):
    experiment.dataset.download()

    outputDir = folder_manager.createTempFolder("qiime_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 1: Demux",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Workspace] Failed to create output dataset")

    if experiment.parameters["barcodeColumn"]:
        demultiplexing(CustomDataset.fetchById(experiment.dataset.id), experiment, outputDataset, outputDir)
    else:
        importDemultiplexedSamples(
            experiment.dataset,
            experiment,
            outputDataset,
            outputDir
        )


if __name__ == "__main__":
    initializeProject(main, SequenceDataset)