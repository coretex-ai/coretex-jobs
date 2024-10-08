from typing import Optional
from pathlib import Path
from zipfile import ZipFile

import logging

from qiime2.sdk.result import Artifact

import qiime2.core.archive as archive

from coretex import CustomDataset, CustomSample, TaskRun, folder_manager, currentTaskRun, createDataset
from coretex.bioinformatics import ctx_qiime2


def isValidArtifact(filePath: Path) -> bool:
    try:
        archiver = archive.Archiver.load(filePath)
    except ValueError:
        return False

    return Artifact._is_valid_type(archiver.type)  # type: ignore


def getQzaPath(sample: CustomSample) -> Optional[Path]:
    zipPath = sample.zipPath

    if isValidArtifact(zipPath):
        qzaPath = folder_manager.temp / f"{zipPath.stem}.qza"
        zipPath.link_to(qzaPath)

        return qzaPath

    qzaPaths = list(sample.path.glob("*.qza"))
    if len(qzaPaths) == 1:
        return qzaPaths[0]

    return None


def importReferenceDataset(dataset: CustomDataset, outputDir: Path, taskRun: TaskRun) -> Path:
    referenceCacheName = f"OTU Reference Dataset Imported to Qiime - {dataset.id}"
    caches = CustomDataset.fetchAll(
        name = referenceCacheName,
        include_sessions = 1
    )

    for cache in caches:
        if cache.count > 0:
            dataset = cache
            break

    if dataset.count > 1:
        raise ValueError(f">> [Qiime: Clustering] Reference dataset must only contain one sample with the OTU fasta file. Found {len(dataset.samples)}")

    dataset.download()
    sample = dataset.samples[0]
    sample.unzip()

    fastaPaths = list(sample.path.glob("*.fasta"))
    qzaPath = getQzaPath(sample)

    # If the input refrerence sequences are not imported, i.e. they are in fasta format,
    # we import them and upload the output as a cache to Coretex
    if len(fastaPaths) == 1 and qzaPath is None:
        fastaPath = fastaPaths[0]
        qzaPath = outputDir / f"{fastaPath.stem}.qza"
        ctx_qiime2.toolsImport("FeatureData[Sequence]", str(fastaPath), str(qzaPath))

        try:
            with createDataset(CustomDataset, referenceCacheName, taskRun.projectId) as referenceCache:
                outputPath = outputDir / "reference-sequences"
                with ZipFile(outputPath, "w") as outputFile:
                    outputFile.write(qzaPath, qzaPath.name)

                referenceCache.add(outputPath)
                return qzaPath
        except BaseException as ex:
            logging.error(f">> [Qiime: Clustering] Failed to create imported reference sequences cache - \"{ex}\"")
            return qzaPath

    if qzaPath is not None and len(fastaPaths) == 0:
        return qzaPath

    raise FileNotFoundError(">> [Qiime: Clustering] Reference dataset must contain a single sample with one file in either .fasta or .qza format")


def closedReferenceClustering(
    sample: CustomSample,
    referenceSequencesPath: Path,
    outputDir: Path,
    percentIdentity: float,
    threads: Optional[int]
) -> Path:

    tablePath = sample.path / "table.qza"
    sequencesPath = sample.path / "rep-seqs.qza"

    clusteredTablePath = outputDir / "clustered-table.qza"
    clusteredSequencesPath = outputDir / "clustered-seqs.qza"
    unmatchedSequencesPath = outputDir / "unmatched-seqs.qza"

    ctx_qiime2.vsearchClusterClosedReference(
        tablePath,
        sequencesPath,
        referenceSequencesPath,
        percentIdentity,
        clusteredTablePath,
        clusteredSequencesPath,
        unmatchedSequencesPath,
        threads
    )

    outputPath = outputDir / "otu.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(clusteredTablePath, clusteredTablePath.name)
        outputFile.write(clusteredSequencesPath, clusteredSequencesPath.name)
        outputFile.write(unmatchedSequencesPath, unmatchedSequencesPath.name)

    return outputPath


def openReferenceClustering(
    sample: CustomSample,
    referenceSequencesPath: Path,
    outputDir: Path,
    percentIdentity: float,
    threads: Optional[int]
) -> Path:

    tablePath = sample.path / "table.qza"
    sequencesPath = sample.path / "rep-seqs.qza"

    clusteredTablePath = outputDir / "clustered-table.qza"
    clusteredSequencesPath = outputDir / "clustered-seqs.qza"
    newReferenceSequencesPath = outputDir / "unmatched-seqs.qza"

    ctx_qiime2.vsearchClusterOpenReference(
        tablePath,
        sequencesPath,
        referenceSequencesPath,
        percentIdentity,
        clusteredTablePath,
        clusteredSequencesPath,
        newReferenceSequencesPath,
        threads
    )

    outputPath = outputDir / "otu.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(clusteredTablePath, clusteredTablePath.name)
        outputFile.write(clusteredSequencesPath, clusteredSequencesPath.name)
        outputFile.write(newReferenceSequencesPath, newReferenceSequencesPath.name)

    return outputPath


def deNovoClustering(
    sample: CustomSample,
    outputDir: Path,
    percentIdentity: float,
    threads: Optional[int]
) -> Path:
    tablePath = sample.path / "table.qza"
    sequencesPath = sample.path / "rep-seqs.qza"

    clusteredTablePath = outputDir / "clustered-table.qza"
    clusteredSequencesPath = outputDir / "clustered-seqs.qza"

    ctx_qiime2.vsearchClusterDeNovo(
        str(tablePath),
        str(sequencesPath),
        percentIdentity,
        str(clusteredTablePath),
        str(clusteredSequencesPath),
        threads
    )

    outputPath = outputDir / "otu.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(clusteredTablePath, clusteredTablePath.name)
        outputFile.write(clusteredSequencesPath, clusteredSequencesPath.name)

    return outputPath


def processSample(
    index: int,
    sample: CustomSample,
    taskRun: TaskRun,
    outputDataset: CustomDataset,
    outputDir: Path,
    clusteringMethod: str
) -> None:

    sample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    percentIdentity = taskRun.parameters["percentIdentity"]
    if percentIdentity <=0 or percentIdentity > 1:
        raise ValueError(">> [Qiime: Clustering] The percent identity parameter must be between 0 and 1.")

    referenceDataset = taskRun.parameters["referenceDataset"]

    threads = taskRun.parameters["threads"]
    if clusteringMethod == "De Novo":
        logging.info(">> [Qiime: Clustering] Performing de novo clustering")
        otuPath = deNovoClustering(
            sample,
            sampleOutputDir,
            taskRun.parameters["percentIdentity"],
            threads
        )
    elif referenceDataset is not None:
        logging.info(">> [Qiime: Clustering] Importing reference dataset")
        referenceSequencesPath = importReferenceDataset(referenceDataset, outputDir, taskRun)
        if clusteringMethod == "Closed Reference":
            logging.info(">> [Qiime: Clustering] Performing closed reference clustering")
            otuPath = closedReferenceClustering(
                sample,
                referenceSequencesPath,
                outputDir,
                percentIdentity,
                threads
            )

        if clusteringMethod == "Open Reference":
            logging.info(">> [Qiime: Clustering] Performing open reference custering")
            otuPath = openReferenceClustering(
                sample,
                referenceSequencesPath,
                outputDir,
                percentIdentity,
                threads
            )
    else:
        raise ValueError(">> [Qiime: Clustering] referenceDataset parameter must not be empty in case of closed or open reference clustering")

    ctx_qiime2.createSample(f"{index}-otu-clusters", outputDataset, otuPath, taskRun, "Step 4: OTU clustering")


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    dataset = taskRun.dataset
    dataset.download()

    denoisedSamples = ctx_qiime2.getDenoisedSamples(dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Qiime: Clustering] Dataset has 0 denoised samples")

    clusteringMethod = taskRun.parameters["clusteringMethod"]

    outputDir = folder_manager.createTempFolder("otu_output")

    outputDatasetName = f"{taskRun.id}-step-4-otu-clustering-{clusteringMethod.replace(' ', '-').lower()}"
    with createDataset(CustomDataset, outputDatasetName, taskRun.projectId) as outputDataset:
        for sample in denoisedSamples:
            index = ctx_qiime2.sampleNumber(sample)
            processSample(index, sample, taskRun, outputDataset, outputDir, clusteringMethod)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
