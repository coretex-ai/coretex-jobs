from pathlib import Path

from zipfile import ZipFile, ZIP_DEFLATED

import logging

from coretex import CustomDataset, TaskRun, CustomSample, folder_manager


def loadIndexed(dataset: CustomDataset) -> list[Path]:
    dataset.download()
    if len(dataset.samples) != 1:
        raise ValueError(">> [Region Separation] The indexed reference dataset should only contain a single sample with everything inside")

    datasetSample = dataset.samples[0]
    datasetSample.unzip()
    samplePath = Path(datasetSample.path)

    referenceDirs: list[Path] = []
    for referenceDir in samplePath.iterdir():
        if referenceDir.name == "__MACOSX":
            continue

        referenceDirs.append(referenceDir)

    return referenceDirs


def clearDirectory(directory: Path) -> None:
    for file in directory.iterdir():
        file.unlink()


def uploadToCoretex(taskRun: TaskRun[CustomDataset], groups: list[Path]) -> None:
    zipOut = Path(folder_manager.createTempFolder("zipOut"))

    dataset = CustomDataset.createDataset(f"{taskRun.id} - Separated Sequences", taskRun.spaceId)
    if dataset is None:
        raise RuntimeError(">> [SequenceSeparation] Failed to create output dataset")

    for group in groups:
        groupZip = zipOut / (group.name + ".zip")
        with ZipFile(groupZip, 'w', ZIP_DEFLATED) as archive:
            for file in group.iterdir():
                archive.write(file, file.name)

        if CustomSample.createCustomSample(groupZip.name, dataset.id, groupZip) is None:
            raise RuntimeError(">> [SequenceSeparation] Failed to upload sample")

    logging.info(f">> [Region Separation] Output files have been uploaded to dataset {dataset.id}: \"{dataset.name}\"")


def prepareGroups(groupNames: list[str], thresholds: list[int], outDir: Path) -> tuple[list[Path], list[int]]:
    if len(groupNames) != len(thresholds) + 1:
        raise ValueError(">> [Region Separation] The number of entered thresholds has to be one less then the number of groups")

    groups: list[Path] = []
    for groupName in groupNames:
        groupPath = outDir / groupName
        groupPath.mkdir(parents = True)
        groups.append(groupPath)

    thresholds.sort()
    thresholds.insert(0, 0)
    thresholds.append(0)

    return groups, thresholds
