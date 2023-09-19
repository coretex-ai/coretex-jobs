from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import Experiment, CustomDataset, CustomSample, folder_manager
from coretex.bioinformatics import sequence_alignment as sa

from .filepaths import SAMTOOLS


def loadData(dataset: CustomDataset) -> list[Path]:
    samPaths: list[Path] = []

    for sample in dataset.samples:
        sample.unzip()

        for file in Path(sample.path).iterdir():
            if file.suffix == ".sam":
                samPaths.append(file)

    if len(samPaths) == 0:
        raise ValueError(">> [Sequence Alignment] No SAM files found")

    return samPaths


def sam2bam(experiment: Experiment[CustomDataset], samDir: Path) -> None:
    bamDir = folder_manager.createTempFolder("BAM")

    bamDataset = CustomDataset.createDataset(
        f"{experiment.id} - Sequence alignment: BAM",
        experiment.projectId
    )

    if bamDataset is None:
        raise RuntimeError(">> [Sequence Alignment] Failed to create coretex dataset")

    logging.info(f">> [Sequence Alignment] Output BAM files will be uploaded to coretex dataset: \"{bamDataset.name}\", ID: {bamDataset.id}")
    logging.info(">> [Sequence Alignment] Starting file conversion to BAM")

    for path in samDir.iterdir():
        outputPath = bamDir / path.name.replace(path.suffix, ".bam")

        sa.sam2bamCommand(Path(SAMTOOLS), path, outputPath)

        zipSam = folder_manager.temp / f"{outputPath.name}.zip"
        with ZipFile(zipSam , "w", ZIP_DEFLATED) as archive:
            archive.write(outputPath, outputPath.name)

        if CustomSample.createCustomSample(outputPath.name, bamDataset.id, zipSam) is None:
            raise RuntimeError(f">> [Sequence Alignment] Failed to upload {outputPath.name} to coretex")

        logging.info(f">> [Sequence Alignment] {outputPath.name} succesfully created")

    logging.info(">> [Sequence Alignment] All files successfuly converted")
