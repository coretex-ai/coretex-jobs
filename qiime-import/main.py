from coretex import Experiment, CustomDataset, SequenceDataset, folder_manager
from coretex.project import initializeProject

from src.multiplexed import importMultiplexed
from src.demultiplexed import importDemultiplexed


def main(experiment: Experiment[SequenceDataset]):
    dataset = experiment.dataset
    dataset.download()

    outputDir = folder_manager.createTempFolder("qiime_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 1: Import",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Workspace] Failed to create output dataset")

    if experiment.parameters["barcodeColumn"]:
        importMultiplexed(
            CustomDataset.fetchById(experiment.dataset.id),
            experiment,
            outputDataset,
            outputDir
        )
    else:
        importDemultiplexed(
            experiment.dataset,
            experiment,
            outputDataset,
            outputDir
        )


if __name__ == "__main__":
    initializeProject(main, SequenceDataset)
