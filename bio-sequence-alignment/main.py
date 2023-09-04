from pathlib import Path

import logging

from coretex import Run, CustomDataset
from coretex.job import initializeJob
from coretex.bioinformatics import sequence_alignment as sa

from src.index import index
from src.sequence_alignment import sequeneAlignment
from src.sam2bam import sam2bam
from src.filepaths import BWA, SAMTOOLS


def main(run: Run[CustomDataset]) -> None:
    sa.chmodX(Path(BWA))
    sa.chmodX(Path(SAMTOOLS))

    logging.info(">> [Sequence Alignment] 1: Index reference genome")
    genomePrefix = index(run)

    logging.info(">> [Sequence Alignment] 2: Sequence alignment")
    samDir = sequeneAlignment(run, genomePrefix)

    if run.parameters["convertToBAM"]:
        logging.info(">> [Sequence Alignment] 3: Convert SAM files to BAM")
        sam2bam(run, samDir)


if __name__ == "__main__":
    initializeJob(main)
