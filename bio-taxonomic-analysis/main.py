from pathlib import Path
from zipfile import ZipFile

from coretex import CustomDataset, CustomSample, Experiment, qiime2 as ctx_qiime2, cache
from coretex.project import initializeProject
from coretex.folder_management import FolderManager
from coretex.qiime2.utils import sampleNumber, createSample, getDenoisedSamples


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

    taxonomySample = createSample(f"{index}-taxonomy", outputDataset.id, taxonomyPath, experiment, "Step 5: Taxonomic Analysis")

    # Second step:
    # Visualize the results
    taxonomySample.download()
    taxonomySample.unzip()

    visualizationPath = sampleOutputDir / "taxonomy.qzv"
    ctx_qiime2.metadataTabulate(
        str(taxonomySample.joinPath("taxonomy.qza")),
        str(visualizationPath)
    )

    createSample(f"{index}-taxonomy-visualization", outputDataset.id, visualizationPath, experiment, "Step 5: Taxonomic Analysis")

    # Third step:
    # View the taxonomic composition of our samples with interactive bar plots
    taxaBarBlotsPath = sampleOutputDir / "taxa-bar-plots.qzv"

    ctx_qiime2.taxaBarplot(
        str(sample.joinPath("table.qza")),
        str(taxonomySample.joinPath("taxonomy.qza")),
        str(importedSample.joinPath(experiment.parameters["barcodesFileName"])),
        str(taxaBarBlotsPath)
    )

    createSample(f"{index}-taxonomy-bar-plots", outputDataset.id, taxaBarBlotsPath, experiment, "Step 5: Taxonomic Analysis")


def main(experiment: Experiment[CustomDataset]):
    experiment.dataset.download()

    denoisedSamples = getDenoisedSamples(experiment.dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Workspace] Dataset has 0 denoised samples")

    importedDataset: CustomDataset = experiment.parameters["importedDataset"]
    importedDataset.download()

    outputDir = Path(FolderManager.instance().createTempFolder("qiime_output"))
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 5: Taxonomic analysis",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Workspace] Failed to create output dataset")

    for sample in denoisedSamples:
        index = sampleNumber(sample)

        importedSample = importedDataset.getSample(f"{index}-import")
        if importedSample is None:
            raise ValueError(f">> [Workspace] Imported sample not found")

        processSample(
            index,
            sample,
            importedSample,
            experiment,
            outputDataset,
            outputDir
        )


if __name__ == "__main__":
    initializeProject(main)
