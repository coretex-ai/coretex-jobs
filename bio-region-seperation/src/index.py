from pathlib import Path

import logging

from coretex import CustomDataset
from coretex.folder_management import FolderManager
from coretex.bioinformatics.sequence_alignment import indexCommand, loadFa

from .filepaths import BWA


def index(dataset: CustomDataset) -> list[Path]:
    referencesDir = Path(FolderManager.instance().createTempFolder("reference"))

    logging.info(">> [Sequence Alignment] Downloading dataset")
    dataset.download()
    referenceSequences = loadFa(dataset)

    for sequence in referenceSequences:
        name = sequence.stem
        indexDir = referencesDir / name
        prefix = indexDir / name

        indexDir.mkdir()
        indexCommand(Path(BWA), sequence, prefix)

    logging.info(">> [Sequence Alignment] Reference sequences succesfully indexed")

    return list(referencesDir.iterdir())
