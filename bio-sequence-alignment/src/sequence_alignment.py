from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import Experiment, CustomDataset, CustomSample, folder_manager

from .utils import alignCommand


def loadData(dataset: CustomDataset) -> list[Path]:
    sequencePaths: list[Path] = []

    for sample in dataset.samples:
        sample.unzip()

        for file in Path(sample.path).iterdir():
            if file.suffix == ".fasta" or file.suffix == ".fastq":
                sequencePaths.append(file)

    if len(sequencePaths) == 0:
        raise ValueError(">> [Sequence Alignment] No sequence reads found")

    return sequencePaths


def sequeneAlignment(experiment: Experiment[CustomDataset], genomePrefix: Path) -> Path:
    samDir = folder_manager.createTempFolder("SAM")

    logging.info(">> [Sequence Alignment] Starting dataset download...")
    experiment.dataset.download()

    logging.info(">> [Sequence Alignment] Dataset downloaded")
    sequencePaths = loadData(experiment.dataset)

    samDataset = CustomDataset.createDataset(
        f"{experiment.id} - Sequence alignment: SAM",
        experiment.spaceId
    )

    if samDataset is None:
        raise RuntimeError(">> [Sequence Alignment] Failed to create coretex dataset")

    logging.info(f">> [Sequence Alignment] Output SAM files will be uploaded to coretex dataset: \"{samDataset.name}\", ID: {samDataset.id}")
    logging.info(">> [Sequence Alignment] Starting sequence alignment process")

    for path in sequencePaths:
        outputPath = samDir / path.name.replace(path.suffix, ".sam")

        alignCommand(genomePrefix, path, outputPath)

        zipSam = folder_manager.temp / f"{outputPath.name}.zip"
        with ZipFile(zipSam , "w", ZIP_DEFLATED) as archive:
            archive.write(outputPath, outputPath.name)

        if CustomSample.createCustomSample(outputPath.name, samDataset.id, zipSam) is None:
            raise RuntimeError(f">> [Sequence Alignment] Failed to upload {outputPath.name} to coretex")

        logging.info(f">> [Sequence Alignment] {outputPath.name} succesfully created")

    logging.info(">> [Sequence Alignment] Sequence alignment finished")

    return samDir
