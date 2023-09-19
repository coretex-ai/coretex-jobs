from coretex import TaskRun, CustomDataset, SequenceDataset, folder_manager, currentTaskRun
from coretex.bioinformatics import ctx_qiime2

from src.multiplexed import importMultiplexed
from src.demultiplexed import importDemultiplexed


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    dataset = taskRun.dataset
    dataset.download()

    outputDir = folder_manager.createTempFolder("import_output")

    multiplexedFastqs = ctx_qiime2.getFastqMPSamples(dataset)
    if len(multiplexedFastqs) > 0:
        importMultiplexed(multiplexedFastqs, taskRun, outputDir)
    else:
        importDemultiplexed(SequenceDataset.decode(dataset.encode()), taskRun, outputDir)


if __name__ == "__main__":
    main()
