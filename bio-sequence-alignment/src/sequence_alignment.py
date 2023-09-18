from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import TaskRun, CustomDataset, CustomSample, folder_manager
from coretex.bioinformatics import sequence_alignment as sa

from .filepaths import BWA


def sequeneAlignment(taskRun: TaskRun[CustomDataset], genomePrefix: Path) -> Path:
    samDir = folder_manager.createTempFolder("SAM")

    logging.info(">> [Sequence Alignment] Starting dataset download...")
    taskRun.dataset.download()

    logging.info(">> [Sequence Alignment] Dataset downloaded")
    sequencePaths = sa.loadFa(taskRun.dataset)

    samDataset = CustomDataset.createDataset(
        f"{taskRun.id} - Sequence alignment: SAM",
        taskRun.spaceId
    )

    if samDataset is None:
        raise RuntimeError(">> [Sequence Alignment] Failed to create coretex dataset")

    logging.info(f">> [Sequence Alignment] Output SAM files will be uploaded to coretex dataset: \"{samDataset.name}\", ID: {samDataset.id}")
    logging.info(">> [Sequence Alignment] Starting sequence alignment process")

    for path in sequencePaths:
        outputPath = samDir / path.name.replace(path.suffix, ".sam")

        sa.alignCommand(Path(BWA), genomePrefix, path, outputPath)

        zipSam = folder_manager.temp / f"{outputPath.name}.zip"
        with ZipFile(zipSam , "w", ZIP_DEFLATED) as archive:
            archive.write(outputPath, outputPath.name)

        if CustomSample.createCustomSample(outputPath.name, samDataset.id, zipSam) is None:
            raise RuntimeError(f">> [Sequence Alignment] Failed to upload {outputPath.name} to coretex")

        logging.info(f">> [Sequence Alignment] {outputPath.name} succesfully created")

    logging.info(">> [Sequence Alignment] Sequence alignment finished")

    return samDir
