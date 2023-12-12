from typing import Optional, List
from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomDataset, CustomSample, TaskRun, folder_manager, createDataset
from coretex.bioinformatics import ctx_qiime2

from .utils import convertMetadata


FORWARD_FASTQ = "forward.fastq"
REVERSE_FASTQ = "reverse.fastq"
BARCODES_FASTQ = "barcodes.fastq"


def importSample(sequencesPath: Path, sequenceType: str, outputDir: Path) -> Path:
    importedSequencesPath = outputDir / "multiplexed-sequences.qza"

    ctx_qiime2.toolsImport(sequenceType, str(sequencesPath), str(importedSequencesPath))

    outputPath = outputDir / "import-output.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(importedSequencesPath, importedSequencesPath.name)

    return outputPath


def importMetadata(metadataPath: Path, outputDir: Path) -> Path:
    metadataPath = convertMetadata(metadataPath)

    outputPath = outputDir / "metadata-output.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(metadataPath, metadataPath.name)

    return outputPath


def prepareFastq(sample: CustomSample, fileName: str, sequenceDir: Path) -> Optional[Path]:
    gzipPath = sample.joinPath(f"{fileName}.gz")
    outputPath = sequenceDir / gzipPath.name

    if gzipPath.exists():
        gzipPath.link_to(outputPath)
        return outputPath

    filePath = sample.joinPath(fileName)
    if filePath.exists():
        ctx_qiime2.compressGzip(filePath, outputPath)
        return outputPath

    return None


def importMultiplexed(
    fastqSamples: List[CustomSample],
    taskRun: TaskRun,
    outputDir: Path
) -> None:

    outputDatasetName = f"{taskRun.id} - Step 1: Import - Multiplexed"
    with createDataset(CustomDataset, outputDatasetName, taskRun.projectId) as outputDataset:

        logging.info(">> [Qiime: Import] Preparing multiplexed data for import into Qiime2")

        for index, sample in enumerate(fastqSamples):
            logging.info(f">> [Qiime: Import] Importing sample {index}")
            sample.unzip()

            sequenceFolderPath = folder_manager.createTempFolder("sequencesFolder")

            barcodesPath = prepareFastq(sample, BARCODES_FASTQ, sequenceFolderPath)
            forwardPath = prepareFastq(sample, FORWARD_FASTQ, sequenceFolderPath)
            reversePath = prepareFastq(sample, REVERSE_FASTQ, sequenceFolderPath)

            metadataPath = sample.path / taskRun.parameters["metadataFileName"]

            if forwardPath is None or barcodesPath is None or not metadataPath.exists():
                raise FileNotFoundError(f">> [Qiime: Import] Each sample must contain one metadata file, {FORWARD_FASTQ}, {BARCODES_FASTQ} and optionaly {REVERSE_FASTQ} in case of paired-end reads. {sample.name} fails to meet these requirements")

            sequenceType = "EMPPairedEndSequences" if reversePath else "EMPSingleEndSequences"

            logging.info(">> [Qiime: Import] Importing sample")
            importedFilePath = importSample(sequenceFolderPath, sequenceType, outputDir)
            logging.info(">> [Qiime: Import] Uploading sample")
            ctx_qiime2.createSample(f"{index}-import", outputDataset.id, importedFilePath, taskRun, "Step 1: Import")

            zippedMetadataPath = importMetadata(metadataPath, outputDir)
            ctx_qiime2.createSample(f"{index}-metadata", outputDataset.id, zippedMetadataPath, taskRun, "Step 1: Import")

    taskRun.submitOutput("outputDataset", outputDataset)
