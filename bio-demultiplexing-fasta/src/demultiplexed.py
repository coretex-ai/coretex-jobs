from typing import List, Tuple
from pathlib import Path
from zipfile import ZipFile

import csv
import logging

from coretex import CustomDataset, CustomSample, Experiment
from coretex.bioinformatics import ctx_qiime2, isPairedEnd

from .utils import summarizeSample


def importSample(inputPath: Path, sequenceType: str, inputFormat: str, outputDir: Path) -> Path:
    demuxPath = outputDir / "demux.qza"

    ctx_qiime2.toolsImport(
        sequenceType,
        str(inputPath),
        str(demuxPath),
        inputFormat
    )

    demuxZipPath = outputDir / "demux.zip"
    with ZipFile(demuxZipPath, "w") as demuxFile:
        demuxFile.write(demuxPath, "demux.qza")

    return demuxZipPath


def importMetadata(metadata: CustomSample, outputDir: Path) -> Path:
    metadataZipPath = outputDir / "metadata.zip"
    with ZipFile(metadataZipPath, "w") as metadataFile:
        metadataFile.write(Path(metadata.path) / "metadata.tsv", "metadata.tsv")

    return metadataZipPath


def createManifestSingle(samples: List[CustomSample], manifestPath: Path) -> Path:
    with manifestPath.open("w") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "absolute-filepath"])

    for sample in samples:
        fastqPath, sampleId = loadSingleEnd(sample)

        with manifestPath.open("a") as manifestFile:
            csv.writer(manifestFile, delimiter = "\t").writerow([sampleId, fastqPath])

    return manifestPath


def createManifestPaired(samples: List[CustomSample], manifestPath: Path) -> Path:
    with manifestPath.open("w") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "forward-absolute-filepath", "reverse-absolute-filepath"])

    for sample in samples:
        forwardPath, reversePath, sampleId = loadPairedEnd(sample)
        with manifestPath.open("a") as manifestFile:
            csv.writer(manifestFile, delimiter = "\t").writerow([sampleId, forwardPath, reversePath])

    return manifestPath


def dumpGzFiles(samples: List[CustomSample], outFolder: Path) ->  None:
    for sample in samples:
        if sample.name.startswith("_metadata"):
            continue

        for filePath in sample.path.glob("*.fastq.gz"):
            filePath.link_to(outFolder / filePath.name)


def loadSingleEnd(sample: CustomSample) -> Tuple[Path, str]:
    sample.unzip()

    filePathList = list(sample.path.glob("*.fastq*"))
    if len(filePathList) == 1:
        filePath = filePathList[0]
        return filePath, filePath.name.split("_")[0]

    raise ValueError(f">> [Microbiome analysis] Sample \"{sample.name}\" must contain exactly one fastq file")


def loadPairedEnd(sample: CustomSample) -> Tuple[Path, Path, str]:
    sample.unzip()

    forwardPathList = list(sample.path.glob("*_R1_*.fastq*"))
    reversePathList = list(sample.path.glob("*_R2_*.fastq*"))

    if len(forwardPathList) != 1 or len(reversePathList) != 1:
        raise ValueError(f">> [Microbiome analysis] Invalid paired-end sample: {sample.name}. Must contain 2 files, one with \"_R1_\" and another with \"_R2_\" in name")

    forwardPath = forwardPathList[0]
    reversePath = reversePathList[0]

    return forwardPath, reversePath, forwardPath.name.split("_")[0]


def isGzCompressed(dataset: CustomDataset) -> bool:
    for sample in dataset.samples:
        sample.unzip()
        if len(list(sample.path.glob("*.fastq.gz"))) > 0:
            return True

    return False


def importDemultiplexedSamples(
    dataset: CustomDataset,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
):

    logging.info(">> [Microbiome analysis] Preparing data for import into QIIME2 format")
    metadata = ctx_qiime2.getMetadataSample(experiment.dataset)

    pairedEnd = isPairedEnd(dataset)
    if isGzCompressed(dataset):
        inputPath = outputDir / "fastqGzFolder"
        inputPath.mkdir()
        dumpGzFiles(dataset.samples, inputPath)
        inputFormat = "CasavaOneEightSingleLanePerSampleDirFmt"

        if pairedEnd:
            sequenceType = "SampleData[PairedEndSequencesWithQuality]"
        else:
            sequenceType = "SampleData[SequencesWithQuality]"
    else:
        inputPath = outputDir / "manifest.tsv"
        fastqSamples = ctx_qiime2.getFastqDPSamples(dataset)

        if pairedEnd:
            createManifestPaired(fastqSamples, inputPath)
            sequenceType = "SampleData[PairedEndSequencesWithQuality]"
            inputFormat = "PairedEndFastqManifestPhred33V2"
        else:
            createManifestSingle(fastqSamples, inputPath)
            sequenceType = "SampleData[SequencesWithQuality]"
            inputFormat = "SingleEndFastqManifestPhred33V2"

    logging.info(">> [Microbiome analysis] Importing data...")
    demuxZipPath = importSample(inputPath, sequenceType, inputFormat, outputDir)
    demuxSample = ctx_qiime2.createSample("0-demux", outputDataset.id, demuxZipPath, experiment, "Step 1: Demultiplexing")

    metadataZipPath = importMetadata(metadata, outputDir)
    ctx_qiime2.createSample("0-import", outputDataset.id, metadataZipPath, experiment, "Step 1: Demultiplexing")

    demuxSample.download()
    demuxSample.unzip()

    logging.info(">> [Microbiome analysis] Creating summarization...")
    visualizationPath = summarizeSample(demuxSample, outputDir)
    ctx_qiime2.createSample("0-summary", outputDataset.id, visualizationPath, experiment, "Step 1: Demultiplexing")
