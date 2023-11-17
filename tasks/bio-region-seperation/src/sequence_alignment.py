from pathlib import Path

import logging

from coretex.bioinformatics import sequence_alignment as sa

from .filepaths import BWA, SAMTOOLS


def alignToRefDatabase(filePath: Path, referenceDir: list[Path], outDir: Path) -> None:
    for dir in referenceDir:
        name = dir.name
        prefix = dir / name
        outputPath = outDir / (name + ".sam")

        sa.alignCommand(Path(BWA), prefix, filePath, outputPath)
        logging.info(f">> [Region Separation] Alignment for {filePath.name} with {name} has been completed")

    logging.info(f">> [Region Separation] All alignments for {filePath.name} have been completed")


def sam2bam(samDir: Path, outDir: Path) -> None:
    for file in samDir.iterdir():
        outFile = outDir / (file.stem + ".bam")

        sa.sam2bamCommand(Path(SAMTOOLS), file, outFile)
        logging.info(f">> [Region Separation] {file.name} converted to bam")
