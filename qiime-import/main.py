from coretex import Run, CustomDataset, SequenceDataset, folder_manager
from coretex.job import initializeJob
from coretex.bioinformatics import ctx_qiime2

from src.multiplexed import importMultiplexed
from src.demultiplexed import importDemultiplexed


def main(run: Run[CustomDataset]):
    dataset = run.dataset
    dataset.download()

    outputDir = folder_manager.createTempFolder("import_output")

    multiplexedFastqs = ctx_qiime2.getFastqMPSamples(dataset)
    if len(multiplexedFastqs) > 0:
        importMultiplexed(multiplexedFastqs, run, outputDir)
    else:
        importDemultiplexed(SequenceDataset.decode(dataset.encode()), run, outputDir)


if __name__ == "__main__":
    initializeJob(main)
