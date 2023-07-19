from pathlib import Path

import logging

from coretex import CustomDataset
from coretex.folder_management import FolderManager
from coretex.bioinformatics import sequence_alignment as sa

from .filepaths import BWA


def index(dataset: CustomDataset) -> list[Path]:
    referencesDir = Path(FolderManager.instance().createTempFolder("reference"))

    logging.info(">> [Region Separation] Downloading dataset")
    dataset.download()
    referenceSequences = sa.loadFa(dataset)

    for sequence in referenceSequences:
        name = sequence.stem
        indexDir = referencesDir / name
        prefix = indexDir / name

        indexDir.mkdir()
        sa.indexCommand(Path(BWA), sequence, prefix)

    logging.info(">> [Region Separation] Reference sequences succesfully indexed")

    return list(referencesDir.iterdir())
