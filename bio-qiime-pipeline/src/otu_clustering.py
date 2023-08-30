from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomDataset, CustomSample, Experiment, folder_manager
from coretex.bioinformatics import ctx_qiime2

from .caching import getCacheNameFour


def importReferenceDataset(dataset: CustomDataset, outputDir: Path, experiment: Experiment) -> Path:
    referenceCacheName = f"OTU Reference Dataset Imported to Qiime - {dataset.id}"
    caches = CustomDataset.fetchAll(queryParameters = [f"name={referenceCacheName}", "include_sessions=1"])
    if len(caches) > 0:
        if caches[0].count > 0:
            dataset = CustomDataset.fetchById(caches[0].id)

    if dataset.count > 1:
        raise ValueError(f">> [Qiime: Clustering] Reference dataset must only contain one sample with the OTU fasta file. Found {len(dataset.samples)}")

    dataset.download()
    sample = dataset.samples[0]
    sample.unzip()

    fastaPaths = list(sample.path.glob("*.fasta"))
    qzaPaths = list(sample.path.glob("*.qza"))

    # If the input refrerence sequences are not imported, i.e. they are in fasta format,
    # we import them and upload the output as a cache to Coretex
    if len(fastaPaths) == 1 and len(qzaPaths) == 0:
        fastaPath = fastaPaths[0]
        qzaPath = outputDir / f"{fastaPath.stem}.qza"
        ctx_qiime2.toolsImport("FeatureData[Sequence]", str(fastaPath), str(qzaPath))

        referenceCache = CustomDataset.createDataset(referenceCacheName, experiment.spaceId)
        if referenceCache is None:
            logging.error(">> [Qiime: Clustering] Failed to create imported reference sequences cache")
            return qzaPath

        outputPath = outputDir / "reference-sequences"
        with ZipFile(outputPath, "w") as outputFile:
            outputFile.write(qzaPath, qzaPath.name)

        if CustomSample.createCustomSample("reference-sequences", referenceCache.id, outputPath) is None:
            logging.error(">> [Qiime: Clustering] Failed to upload imported reference sequences cache")

        return qzaPath

    if len(qzaPaths) == 1 and len(fastaPaths) == 0:
        return qzaPaths[0]

    raise FileNotFoundError(">> [Qiime: Clustering] Reference dataset must contain a single sample with one file in either .fasta or .qza format")


def closedReferenceClustering(
    sample: CustomSample,
    referenceSequencesPath: Path,
    outputDir: Path,
    percentIdentity: float,
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
        unmatchedSequencesPath
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
        newReferenceSequencesPath
    )

    outputPath = outputDir / "otu.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(clusteredTablePath, clusteredTablePath.name)
        outputFile.write(clusteredSequencesPath, clusteredSequencesPath.name)
        outputFile.write(newReferenceSequencesPath, newReferenceSequencesPath.name)

    return outputPath


def deNovoClustering(sample: CustomSample, outputDir: Path, percentIdentity: float) -> Path:
    tablePath = sample.path / "table.qza"
    sequencesPath = sample.path / "rep-seqs.qza"

    clusteredTablePath = outputDir / "clustered-table.qza"
    clusteredSequencesPath = outputDir / "clustered-seqs.qza"

    ctx_qiime2.vsearchClusterDeNovo(
        str(tablePath),
        str(sequencesPath),
        percentIdentity,
        str(clusteredTablePath),
        str(clusteredSequencesPath)
    )

    outputPath = outputDir / "otu.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(clusteredTablePath, clusteredTablePath.name)
        outputFile.write(clusteredSequencesPath, clusteredSequencesPath.name)

    return outputPath


def processSample(
    index: int,
    sample: CustomSample,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path,
    clusteringMethod: str
) -> None:

    sample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    percentIdentity = experiment.parameters["percentIdentity"]
    if percentIdentity <=0 or percentIdentity > 1:
        raise ValueError(">> [Qiime: Clustering] The percent identity parameter must be between 0 and 1.")

    referenceDataset = experiment.parameters["referenceDataset"]

    if clusteringMethod == "De Novo":
        logging.info(">> [Qiime: Clustering] Performing de novo clustering")
        otuPath = deNovoClustering(sample, sampleOutputDir, experiment.parameters["percentIdentity"])
    elif referenceDataset is not None:
        logging.info(">> [Qiime: Clustering] Importing reference dataset")
        referenceSequencesPath = importReferenceDataset(referenceDataset, outputDir, experiment)
        if clusteringMethod == "Closed Reference":
            logging.info(">> [Qiime: Clustering] Performing closed reference clustering")
            otuPath = closedReferenceClustering(sample, referenceSequencesPath, outputDir, percentIdentity)
        else:  # Open Reference
            logging.info(">> [Qiime: Clustering] Performing open reference custering")
            otuPath = openReferenceClustering(sample, referenceSequencesPath, outputDir, percentIdentity)
    else:
        raise ValueError(">> [Qiime: Clustering] referenceDataset parameter must not be empty in case of closed or open reference clustering")

    ctx_qiime2.createSample(f"{index}-otu-clusters", outputDataset.id, otuPath, experiment, "Step 4: OTU clustering")


def otuClustering(
    dataset: CustomDataset,
    experiment: Experiment
) -> CustomDataset:

    denoisedSamples = ctx_qiime2.getDenoisedSamples(dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Qiime: Clustering] Dataset has 0 denoised samples")

    clusteringMethod = experiment.parameters["clusteringMethod"]

    outputDir = folder_manager.createTempFolder("otu_output")
    outputDataset = CustomDataset.createDataset(
        getCacheNameFour(experiment),
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Qiime: Clustering] Failed to create output dataset")

    for sample in denoisedSamples:
        index = ctx_qiime2.sampleNumber(sample)
        processSample(index, sample, experiment, outputDataset, outputDir, clusteringMethod)

    outputDataset.refresh()
    return outputDataset
