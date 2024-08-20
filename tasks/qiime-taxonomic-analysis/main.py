from typing import Optional
from pathlib import Path
from zipfile import ZipFile

from coretex import CustomDataset, CustomSample, TaskRun, cache, folder_manager, currentTaskRun, createDataset
from coretex.bioinformatics import ctx_qiime2


def featureClassifierClassifySklearnSample(
    sample: CustomSample,
    classifierPath: Path,
    outputDir: Path,
    threads: Optional[int]
) -> Path:

    outputPath = outputDir / "taxonomy.qza"

    ctx_qiime2.featureClassifierClassifySklearn(
        str(classifierPath),
        str(sample.joinPath("rep-seqs.qza")),
        str(outputPath),
        threads
    )

    outputZipPath = outputDir / "taxonomy.zip"
    with ZipFile(outputZipPath, "w") as outputFile:
        outputFile.write(outputPath, outputPath.name)

    return outputZipPath


def processSample(
    index: int,
    sample: CustomSample,
    metadataSample: CustomSample,
    taskRun: TaskRun,
    outputDataset: CustomDataset,
    outputDir: Path,
    threads: Optional[int]
) -> None:

    sample.unzip()
    metadataSample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    # First step:
    # Assign taxonomy to the sequences in our FeatureData[Sequence] QIIME 2 artifact

    if not cache.exists(taskRun.parameters["classifier"]):
        cache.storeUrl(taskRun.parameters["classifier"], taskRun.parameters["classifier"])

    classifierPath = cache.getPath(taskRun.parameters["classifier"])
    classifierPath = classifierPath.rename(classifierPath.parent / f"{classifierPath.stem}.qza")

    taxonomyPath = featureClassifierClassifySklearnSample(
        sample,
        classifierPath,
        sampleOutputDir,
        threads
    )

    taxonomySample = ctx_qiime2.createSample(f"{index}-taxonomy", outputDataset, taxonomyPath, taskRun, "Step 5: Taxonomic Analysis")

    # Second step:
    # Visualize the results
    taxonomySample.download()
    taxonomySample.unzip()

    visualizationPath = sampleOutputDir / "taxonomy.qzv"
    ctx_qiime2.metadataTabulate(
        str(taxonomySample.joinPath("taxonomy.qza")),
        str(visualizationPath)
    )

    ctx_qiime2.createSample(f"{index}-taxonomy-visualization", outputDataset, visualizationPath, taskRun, "Step 5: Taxonomic Analysis")

    # Third step:
    # View the taxonomic composition of our samples with interactive bar plots
    taxaBarBlotsPath = sampleOutputDir / "taxa-bar-plots.qzv"

    ctx_qiime2.taxaBarplot(
        str(sample.joinPath("table.qza")),
        str(taxonomySample.joinPath("taxonomy.qza")),
        str(ctx_qiime2.getMetadata(metadataSample)),
        str(taxaBarBlotsPath)
    )

    ctx_qiime2.createSample(f"{index}-taxonomy-bar-plots", outputDataset, taxaBarBlotsPath, taskRun, "Step 5: Taxonomic Analysis")


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    taskRun.dataset.download()

    denoisedSamples = ctx_qiime2.getDenoisedSamples(taskRun.dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Qiime: Taxonomic Analysis] Dataset has 0 denoised samples")

    importedDataset: CustomDataset = taskRun.parameters["importedDataset"]
    importedDataset.download()

    outputDir = folder_manager.createTempFolder("taxonomy_output")

    outputDatasetName = f"{taskRun.id}-step-5-taxonomic-analysis"
    with createDataset(CustomDataset, outputDatasetName, taskRun.projectId) as outputDataset:
        for sample in denoisedSamples:
            index = ctx_qiime2.sampleNumber(sample)

            metadataSample = importedDataset.getSample(f"{index}-metadata")
            if metadataSample is None:
                raise ValueError(f">> [Qiime: Taxonomic Analysis] Imported sample not found")

            processSample(
                index,
                sample,
                metadataSample,
                taskRun,
                outputDataset,
                outputDir,
                taskRun.parameters["threads"]
            )

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
