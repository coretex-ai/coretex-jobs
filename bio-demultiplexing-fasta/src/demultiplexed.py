from typing import List
from pathlib import Path
from zipfile import ZipFile

import csv

from coretex import CustomDataset, CustomSample, Experiment, qiime2 as ctx_qiime2
from coretex.qiime2.utils import createSample

from .utils import summarizeSample


def importSample(manifestPath: Path, sequenceType: str, outputDir: Path) -> Path:
    demuxPath = outputDir / "demux.qza"

    ctx_qiime2.toolsImport(
        "SampleData[SequencesWithQuality]",
        str(manifestPath),
        str(demuxPath),
        sequenceType
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


def createManifest(samples: List[CustomSample], manifestPath: Path) -> Path:
    with manifestPath.open("w") as manifestFile:
        csv.writer(manifestFile, delimiter = "\t").writerow(["sample-id", "absolute-filepath"])

    for sample in samples:
        for fastqPath in Path(sample.path).iterdir():
            if fastqPath.suffix != ".fastq":
                continue

            sampleId = fastqPath.name[:-6]
            with manifestPath.open("a") as manifestFile:
                csv.writer(manifestFile, delimiter = "\t").writerow([sampleId, fastqPath])

    return manifestPath


def importDemultiplexedSamples(
    fastqSamples: List[CustomSample],
    metadata: CustomSample,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
):

    manifestPath = outputDir / "manifest.tsv"
    createManifest(fastqSamples, manifestPath)

    demuxZipPath = importSample(manifestPath, experiment.parameters["sequenceType"], outputDir)
    demuxSample = createSample("0-demux", outputDataset.id, demuxZipPath, experiment, "Step 1: Demultiplexing")

    metadataZipPath = importMetadata(metadata, outputDir)
    createSample("0-import", outputDataset.id, metadataZipPath, experiment, "Step 1: Demultiplexing")

    demuxSample.download()
    demuxSample.unzip()

    visualizationPath = summarizeSample(demuxSample, outputDir)
    createSample("0-summary", outputDataset.id, visualizationPath, experiment, "Step 1: Demultiplexing")
