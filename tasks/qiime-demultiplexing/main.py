from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomSample, CustomDataset, TaskRun, folder_manager, currentTaskRun, createDataset
from coretex.bioinformatics import ctx_qiime2


def forwardMetadata(sample: CustomSample, index: int,  outputDatasetId: int, taskRun: TaskRun) -> Path:
    ctx_qiime2.createSample(f"{index}-metadata", outputDatasetId, sample.zipPath, taskRun, "Step 2: Demultiplexing")
    return ctx_qiime2.getMetadata(sample)


def demuxSummarize(sample: CustomSample, outputDir: Path) -> Path:
    demuxPath = sample.path / "demux.qza"
    visualizationPath = outputDir / "demux.qzv"

    ctx_qiime2.demuxSummarize(str(demuxPath), str(visualizationPath))
    return visualizationPath


def demuxEmpSample(sample: CustomSample, barcodesPath: Path, barcodeColumn: str, outputDir: Path, pairedEnd: bool) -> Path:
    sequencesPath = sample.path / "multiplexed-sequences.qza"

    demuxFilePath = outputDir / "demux.qza"
    demuxDetailsFilePath = outputDir / "demux-details.qza"

    if not pairedEnd:
        ctx_qiime2.demuxEmpSingle(
            str(sequencesPath),
            str(barcodesPath),
            barcodeColumn,
            str(demuxFilePath),
            str(demuxDetailsFilePath)
        )
    else:
        ctx_qiime2.demuxEmpPaired(
            str(sequencesPath),
            str(barcodesPath),
            barcodeColumn,
            str(demuxFilePath),
            str(demuxDetailsFilePath)
        )

    demuxOutputPath = outputDir / "demux-output.zip"

    with ZipFile(demuxOutputPath, "w") as demuxOutput:
        demuxOutput.write(demuxFilePath, demuxFilePath.name)
        demuxOutput.write(demuxDetailsFilePath, demuxDetailsFilePath.name)

    return demuxOutputPath


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    dataset = taskRun.dataset
    dataset.download()

    importedSamples = ctx_qiime2.getImportedSamples(dataset)
    if len(importedSamples) == 0:
        if len(ctx_qiime2.getDemuxSamples(dataset)) > 0:
            logging.info(">> [Qiime: Demux] Forwarding demultiplexed samples")
            taskRun.submitOutput("outputDataset", dataset)
            return None

        raise ValueError(">> [Qiime: Demux] Dataset has 0 imported samples")

    outputDir = folder_manager.createTempFolder("demux_output")

    outputDatasetName = f"{taskRun.id} - Step 2: Demultiplexing"
    with createDataset(CustomDataset, outputDatasetName, taskRun.projectId) as outputDataset:
        for sample in importedSamples:
            sample.unzip()

            index = ctx_qiime2.sampleNumber(sample)

            metadataSample = dataset.getSample(f"{index}-metadata")
            if metadataSample is None:
                raise ValueError(f">> [Qiime: Demux] Metadata sample not found")

            metadataPath = forwardMetadata(metadataSample, index, outputDataset.id, taskRun)
            demuxPath = demuxEmpSample(
                sample,
                metadataPath,
                taskRun.parameters["barcodeColumn"],
                outputDir,
                ctx_qiime2.isPairedEnd(sample)
            )

            demuxSample = ctx_qiime2.createSample(f"{index}-demux", outputDataset.id, demuxPath, taskRun, "Step 2: Demultiplexing")

            demuxSample.download()
            demuxSample.unzip()

            logging.info(">> [Qiime: Demux] Creating summarization")
            visualizationPath = demuxSummarize(demuxSample, outputDir)
            ctx_qiime2.createSample(f"{index}-summary", outputDataset.id, visualizationPath, taskRun, "Step 2: Demultiplexing")

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
