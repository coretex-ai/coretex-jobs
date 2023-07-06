from pathlib import Path

import logging

from coretex import Experiment, CustomDataset
from coretex.project import initializeProject

from src.index import index
from src.sequence_alignment import sequeneAlignment
from src.sam2bam import sam2bam
from src.utils import chmodX
from src.filepaths import BWA, SAMTOOLS


def main(experiment: Experiment[CustomDataset]) -> None:
    chmodX(Path(BWA))
    chmodX(Path(SAMTOOLS))

    logging.info(">> [Sequence Alignment] 1: Index reference genome")
    genomePrefix = index(experiment)

    logging.info(">> [Sequence Alignment] 2: Sequence alignment")
    samDir = sequeneAlignment(experiment, genomePrefix)

    if experiment.parameters["convertToBAM"]:
        logging.info(">> [Sequence Alignment] 3: Convert SAM files to BAM")
        sam2bam(experiment, samDir)


if __name__ == "__main__":
    initializeProject(main)
