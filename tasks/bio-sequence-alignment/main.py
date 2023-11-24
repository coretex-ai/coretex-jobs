from pathlib import Path

import logging

from coretex import TaskRun, CustomDataset, currentTaskRun
from coretex.bioinformatics import sequence_alignment as sa

from src.index import index
from src.sequence_alignment import sequeneAlignment
from src.sam2bam import sam2bam
from src.filepaths import BWA, SAMTOOLS


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    if not taskRun.parameters["genomeUrl"] and not taskRun.parameters["referenceDataset"]:
        raise ValueError(">> [Sequence Alignment] It is required to enter either the genomeUrl or the referenceDataset parameter. Both cannot be null")

    sa.chmodX(Path(BWA))
    sa.chmodX(Path(SAMTOOLS))

    logging.info(">> [Sequence Alignment] 1: Index reference genome")
    genomePrefix = index(taskRun)

    logging.info(">> [Sequence Alignment] 2: Sequence alignment")
    samDir = sequeneAlignment(taskRun, genomePrefix)

    if taskRun.parameters["convertToBAM"]:
        logging.info(">> [Sequence Alignment] 3: Convert SAM files to BAM")
        sam2bam(taskRun, samDir)


if __name__ == "__main__":
    main()
