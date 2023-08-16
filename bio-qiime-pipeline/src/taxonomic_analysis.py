from pathlib import Path
from zipfile import ZipFile

from coretex import CustomDataset, CustomSample, Experiment, cache, folder_manager
from coretex.bioinformatics import ctx_qiime2

from .utils import getMetadata


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
    importedSample: CustomSample,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
):

    sample.unzip()
    importedSample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    # First step:
    # Assign taxonomy to the sequences in our FeatureData[Sequence] QIIME 2 artifact

    # TODO: Do not zip cached samples
    if not cache.exists(experiment.parameters["classifier"]):
        cache.storeUrl(experiment.parameters["classifier"], "classifier.zip")

    classifierPath = cache.getPath(experiment.parameters["classifier"])
    classifierPath = classifierPath.rename(classifierPath.parent / f"{classifierPath.stem}.qza")

    taxonomyPath = featureClassifierClassifySklearnSample(
        sample,
        classifierPath,
        sampleOutputDir
    )

    taxonomySample = ctx_qiime2.createSample(f"{index}-taxonomy", outputDataset.id, taxonomyPath, experiment, "Step 5: Taxonomic Analysis")

    # Second step:
    # Visualize the results
    taxonomySample.download()
    taxonomySample.unzip()

    visualizationPath = sampleOutputDir / "taxonomy.qzv"
    ctx_qiime2.metadataTabulate(
        str(taxonomySample.joinPath("taxonomy.qza")),
        str(visualizationPath)
    )

    ctx_qiime2.createSample(f"{index}-taxonomy-visualization", outputDataset.id, visualizationPath, experiment, "Step 5: Taxonomic Analysis")

    # Third step:
    # View the taxonomic composition of our samples with interactive bar plots
    taxaBarBlotsPath = sampleOutputDir / "taxa-bar-plots.qzv"

    ctx_qiime2.taxaBarplot(
        str(sample.joinPath("table.qza")),
        str(taxonomySample.joinPath("taxonomy.qza")),
        str(getMetadata(importedSample, experiment.parameters["metadataFileName"])),
        str(taxaBarBlotsPath)
    )

    ctx_qiime2.createSample(f"{index}-taxonomy-bar-plots", outputDataset.id, taxaBarBlotsPath, experiment, "Step 5: Taxonomic Analysis")


def taxonomicAnalysis(
    importedDataset: CustomDataset,
    denoisedDataset: CustomDataset,
    experiment: Experiment
) -> None:

    denoisedSamples = ctx_qiime2.getDenoisedSamples(denoisedDataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Microbiome analysis] Dataset has 0 denoised samples")

    outputDir = folder_manager.createTempFolder("taxonomic_output")
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 5: Taxonomic analysis",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Microbiome analysis] Failed to create output dataset")

    for sample in denoisedSamples:
        index = ctx_qiime2.sampleNumber(sample)

        importedSample = importedDataset.getSample(f"{index}-import")
        if importedSample is None:
            raise ValueError(f">> [Microbiome analysis] Imported sample not found")

        processSample(
            index,
            sample,
            importedSample,
            experiment,
            outputDataset,
            outputDir
        )
