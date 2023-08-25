from typing import Optional
from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomDataset, CustomSample, Experiment, folder_manager
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


def getFastq(sample: CustomSample, fileName: str) -> Optional[Path]:
    foundFiles = list(sample.path.glob(f"*{fileName}"))
    foundFiles.extend(list(sample.path.glob(f"*{fileName}.gz")))

    if len(foundFiles) > 1:
        raise RuntimeError(f">> [Qiime Import] Found multiple {fileName} files")

    return foundFiles[0] if len(foundFiles) == 1 else None


def prepareSequences(
    barcodesPath: Path,
    forwardPath: Path,
    reversePath: Optional[Path]
) -> None:

    sequenceFolderPath = folder_manager.createTempFolder("sequencesFolder")

    newBarcodesPath = sequenceFolderPath / f"{BARCODES_FASTQ}.gz"
    newForwardPath = sequenceFolderPath / f"{FORWARD_FASTQ}.gz"
    newReversePath = sequenceFolderPath / f"{REVERSE_FASTQ}.gz"

    if reversePath is None:
        checkGz = [path.suffix == ".gz" for path in [barcodesPath, forwardPath]]
    else:
        checkGz = [path.suffix == ".gz" for path in [barcodesPath, forwardPath, reversePath]]

    if all(checkGz):
        barcodesPath.link_to(newBarcodesPath)
        forwardPath.link_to(newForwardPath)

        if reversePath is not None:
            reversePath.link_to(newReversePath)

        return sequenceFolderPath
    elif not any(checkGz):
        ctx_qiime2.compressGzip(barcodesPath, newBarcodesPath)
        ctx_qiime2.compressGzip(forwardPath, newForwardPath)

        if reversePath is not None:
            ctx_qiime2.compressGzip(reversePath, newReversePath)

        return sequenceFolderPath

    raise RuntimeError(">> [Qiime Import] All fastq files must be either gz compressed or not compressed. Found a mix of both")


def importMultiplexed(
    dataset: CustomDataset,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
) -> None:

    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 1: Import - Multiplexed",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Qiime Import] Failed to create output dataset")

    logging.info(">> [Qiime Import] Preparing multiplexed data for import into Qiime2")

    fastqSamples = ctx_qiime2.getFastqMPSamples(dataset)
    for index, sample in enumerate(fastqSamples):
        logging.info(f">> [Qiime Import] Importing sample {index}")
        sample.unzip()

        barcodesPath = getFastq(sample, BARCODES_FASTQ)
        forwardPath = getFastq(sample, FORWARD_FASTQ)
        reversePath = getFastq(sample, REVERSE_FASTQ)

        metadataPath = sample.path / experiment.parameters["metadataFileName"]

        if forwardPath is None or barcodesPath is None or not metadataPath.exists():
            raise FileNotFoundError(f">> [Qiime Import] Each sample must contain one metadata file, {FORWARD_FASTQ}, {BARCODES_FASTQ} and optionaly {REVERSE_FASTQ} in case of paired-end reads. {sample.name} fails to meet these requirements")

        sequenceFolderPath = prepareSequences(barcodesPath, forwardPath, reversePath)
        sequenceType = "EMPPairedEndSequences" if reversePath else "EMPSingleEndSequences"

        logging.info(">> [Qiime Impot] Importing sample")
        importedFilePath = importSample(sequenceFolderPath, sequenceType, outputDir)
        logging.info(">> [Qiime Impot] Uploading sample")
        ctx_qiime2.createSample(f"{index}-import", outputDataset.id, importedFilePath, experiment, "Step 1: Import")

        zippedMetadataPath = importMetadata(metadataPath, outputDir)
        ctx_qiime2.createSample(f"{index}-metadata", outputDataset.id, zippedMetadataPath, experiment, "Step 1: Import")
