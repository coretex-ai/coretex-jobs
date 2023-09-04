from pathlib import Path
from zipfile import ZipFile

from coretex import CustomDataset, CustomSample, Run, cache, folder_manager
from coretex.job import initializeJob
from coretex.bioinformatics import ctx_qiime2


def featureClassifierClassifySklearnSample(
    sample: CustomSample,
    classifierPath: Path,
    outputDir: Path
) -> Path:

    outputPath = outputDir / "taxonomy.qza"

    ctx_qiime2.featureClassifierClassifySklearn(
        str(classifierPath),
        str(sample.joinPath("rep-seqs.qza")),
        str(outputPath)
    )

    outputZipPath = outputDir / "taxonomy.zip"
    with ZipFile(outputZipPath, "w") as outputFile:
        outputFile.write(outputPath, outputPath.name)

    return outputZipPath


def processSample(
    index: int,
    sample: CustomSample,
    metadataSample: CustomSample,
    run: Run,
    outputDataset: CustomDataset,
    outputDir: Path
):

    sample.unzip()
    metadataSample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    # First step:
    # Assign taxonomy to the sequences in our FeatureData[Sequence] QIIME 2 artifact

    # TODO: Do not zip cached samples
    if not cache.exists(run.parameters["classifier"]):
        cache.storeUrl(run.parameters["classifier"], "classifier.zip")

    classifierPath = cache.getPath(run.parameters["classifier"])
    classifierPath = classifierPath.rename(classifierPath.parent / f"{classifierPath.stem}.qza")

    taxonomyPath = featureClassifierClassifySklearnSample(
        sample,
        classifierPath,
        sampleOutputDir
    )

    taxonomySample = ctx_qiime2.createSample(f"{index}-taxonomy", outputDataset.id, taxonomyPath, run, "Step 5: Taxonomic Analysis")

    # Second step:
    # Visualize the results
    taxonomySample.download()
    taxonomySample.unzip()

    visualizationPath = sampleOutputDir / "taxonomy.qzv"
    ctx_qiime2.metadataTabulate(
        str(taxonomySample.joinPath("taxonomy.qza")),
        str(visualizationPath)
    )

    ctx_qiime2.createSample(f"{index}-taxonomy-visualization", outputDataset.id, visualizationPath, run, "Step 5: Taxonomic Analysis")

    # Third step:
    # View the taxonomic composition of our samples with interactive bar plots
    taxaBarBlotsPath = sampleOutputDir / "taxa-bar-plots.qzv"

    ctx_qiime2.taxaBarplot(
        str(sample.joinPath("table.qza")),
        str(taxonomySample.joinPath("taxonomy.qza")),
        str(ctx_qiime2.getMetadata(metadataSample)),
        str(taxaBarBlotsPath)
    )

    ctx_qiime2.createSample(f"{index}-taxonomy-bar-plots", outputDataset.id, taxaBarBlotsPath, run, "Step 5: Taxonomic Analysis")


def main(run: Run[CustomDataset]):
    run.dataset.download()

    denoisedSamples = ctx_qiime2.getDenoisedSamples(run.dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Qiime: Taxonomic Analysis] Dataset has 0 denoised samples")

    importedDataset: CustomDataset = run.parameters["importedDataset"]
    importedDataset.download()

    outputDir = folder_manager.createTempFolder("taxonomy_output")
    outputDataset = CustomDataset.createDataset(
        f"{run.id} - Step 5: Taxonomic analysis",
        run.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Qiime: Taxonomic Analysis] Failed to create output dataset")

    for sample in denoisedSamples:
        index = ctx_qiime2.sampleNumber(sample)

        metadataSample = importedDataset.getSample(f"{index}-metadata")
        if metadataSample is None:
            raise ValueError(f">> [Qiime: Taxonomic Analysis] Imported sample not found")

        processSample(
            index,
            sample,
            metadataSample,
            run,
            outputDataset,
            outputDir
        )


if __name__ == "__main__":
    initializeJob(main)
