from coretex import Experiment, CustomDataset, SequenceDataset, folder_manager
from coretex.project import initializeProject
from coretex.bioinformatics import ctx_qiime2

from src.multiplexed import importMultiplexed
from src.demultiplexed import importDemultiplexed


def main(experiment: Experiment[CustomDataset]):
    dataset = experiment.dataset
    dataset.download()

    outputDir = folder_manager.createTempFolder("import_output")

    multiplexedFastqs = ctx_qiime2.getFastqMPSamples(dataset)
    if len(multiplexedFastqs) > 0:
        importMultiplexed(multiplexedFastqs, experiment, outputDir)
    else:
        importDemultiplexed(SequenceDataset.decode(dataset.encode()), experiment, outputDir)


if __name__ == "__main__":
    initializeProject(main)
