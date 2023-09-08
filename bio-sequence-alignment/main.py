from pathlib import Path

import logging

from coretex import Experiment, CustomDataset, currentExperiment
from coretex.bioinformatics import sequence_alignment as sa

from src.index import index
from src.sequence_alignment import sequeneAlignment
from src.sam2bam import sam2bam
from src.filepaths import BWA, SAMTOOLS


def main() -> None:
    experiment: Experiment[CustomDataset] = currentExperiment()

    sa.chmodX(Path(BWA))
    sa.chmodX(Path(SAMTOOLS))

    logging.info(">> [Sequence Alignment] 1: Index reference genome")
    genomePrefix = index(experiment)

    logging.info(">> [Sequence Alignment] 2: Sequence alignment")
    samDir = sequeneAlignment(experiment, genomePrefix)

    if experiment.parameters["convertToBAM"]:
        logging.info(">> [Sequence Alignment] 3: Convert SAM files to BAM")
        sam2bam(experiment, samDir)


if __name__ == "__main__":
    main()
