from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomDataset, Experiment, folder_manager
from coretex.bioinformatics import ctx_qiime2

from .utils import convertMetadata


def importSample(sequencesPath: Path, metadataPath: Path, sequenceType: str, outputDir: Path) -> Path:
    importedSequencesPath = outputDir / "multiplexedSequences.qza"

    ctx_qiime2.toolsImport(sequenceType, str(sequencesPath), str(importedSequencesPath))
    metadataPath = convertMetadata(metadataPath)

    outputPath = outputDir / "import-output.zip"

    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(metadataPath, metadataPath.name)
        outputFile.write(importedSequencesPath, importedSequencesPath.name)

    return outputPath


def importMultiplexed(
    dataset: CustomDataset,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
) -> None:

    logging.info(">> [Microbiome analysis] Preparing multiplexed data for import into Qiime2")

    sequenceFolderPath = folder_manager.createTempFolder("data_dir")
    sequencesPath = sequenceFolderPath / "sequences.fastq"
    barcodesPath = sequenceFolderPath / "barcodes.fastq"

    fastqSamples = ctx_qiime2.getFastqMPSamples(dataset)
    for sample in fastqSamples:
        sample.unzip()

        samplePath = Path(sample.path)
        sampleSequences = samplePath / "sequences" / "sequences.fastq"
        sampleBarcodes = samplePath / "sequences" / "barcodes.fastq"

        with sequencesPath.open("a") as sequencesFile:
            sequencesFile.write(sampleSequences.read_text())

        with barcodesPath.open("a") as barcodesFile:
            barcodesFile.write(sampleBarcodes.read_text())

        metadataPath = dataset.samples[0].path / experiment.parameters["metadataFileName"]

        sequencesPath = sequenceFolderPath / "sequences.fastq.gz"
        barcodesPath = sequenceFolderPath / "barcodes.fastq.gz"

        if not sequencesPath.exists():
            source = sequenceFolderPath / "sequences.fastq"
            ctx_qiime2.compressGzip(source, sequencesPath)

        if not barcodesPath.exists():
            source = sequenceFolderPath / "barcodes.fastq"
            ctx_qiime2.compressGzip(source, barcodesPath)

        logging.info(">> [Microbiome analysis] Importing sample")
        importedFilePath = importSample(sequenceFolderPath, metadataPath, experiment.parameters["sequenceType"], outputDir)
        ctx_qiime2.createSample("0-import", outputDataset.id, importedFilePath, experiment, "Step 1: Import")
