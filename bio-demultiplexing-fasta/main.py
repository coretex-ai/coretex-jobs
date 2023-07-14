from pathlib import Path

from coretex import CustomDataset, Experiment
from coretex.project import initializeProject
from coretex.folder_management import FolderManager
from coretex.bioinformatics.qiime2.utils import getFastqMPSamples, getFastqDPSamples, getMetadataSample

from src.multiplexed import demultiplexing
from src.demultiplexed import importDemultiplexedSamples


def main(experiment: Experiment[CustomDataset]):
    experiment.dataset.download()
    multiplexed = True

    fastqSamples = getFastqMPSamples(experiment.dataset)
    if len(fastqSamples) == 0:
        fastqSamples = getFastqDPSamples(experiment.dataset)
        multiplexed = False
        if len(fastqSamples) == 0:
            raise ValueError(">> [Workspace] Dataset has 0 fastq samples")

    outputDir = Path(FolderManager.instance().createTempFolder("qiime_output"))
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 1: Demux",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Workspace] Failed to create output dataset")

    if multiplexed:
        demultiplexing(fastqSamples, experiment, outputDataset, outputDir)
    else:
        metadataSample = getMetadataSample(experiment.dataset)
        importDemultiplexedSamples(fastqSamples, metadataSample, experiment, outputDataset, outputDir)


if __name__ == "__main__":
    initializeProject(main)
