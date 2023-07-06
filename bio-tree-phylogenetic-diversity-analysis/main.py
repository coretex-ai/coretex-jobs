from pathlib import Path
from zipfile import ZipFile

from coretex import CustomDataset, CustomSample, Experiment, qiime2 as ctx_qiime2
from coretex.project import initializeProject
from coretex.folder_management import FolderManager
from coretex.qiime2.utils import sampleNumber, createSample, getDenoisedSamples


def phylogenyAlignToTreeMafftFasttreeSample(sample: CustomSample, outputDir: Path) -> Path:
    sequencesPath = Path(sample.path) / "rep-seqs.qza"

    aligmentPath = outputDir / "aligned-rep-seqs.qza"
    maskedAligmentPath = outputDir / "masked-aligned-rep-seqs.qza"
    unrootedTreePath = outputDir / "unrooted-tree.qza"
    rootedTreePath = outputDir / "rooted-tree.qza"

    ctx_qiime2.phylogenyAlignToTreeMafftFasttree(
        str(sequencesPath),
        str(aligmentPath),
        str(maskedAligmentPath),
        str(unrootedTreePath),
        str(rootedTreePath)
    )

    outputPath = outputDir / "tree.zip"
    with ZipFile(outputPath, "w") as outputFile:
        outputFile.write(aligmentPath, aligmentPath.name)
        outputFile.write(maskedAligmentPath, maskedAligmentPath.name)
        outputFile.write(unrootedTreePath, unrootedTreePath.name)
        outputFile.write(rootedTreePath, rootedTreePath.name)

    return outputPath


def processSample(
    index: int,
    sample: CustomSample,
    experiment: Experiment,
    outputDataset: CustomDataset,
    outputDir: Path
):

    sample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    # Phylogenetic diversity analysis
    treePath = phylogenyAlignToTreeMafftFasttreeSample(sample, sampleOutputDir)
    createSample(f"{index}-phylogenetic-tree", outputDataset.id, treePath, experiment, "Step 3: Phylogenetic tree")


def main(experiment: Experiment[CustomDataset]):
    experiment.dataset.download()

    denoisedSamples = getDenoisedSamples(experiment.dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Workspace] Dataset has 0 denoised samples")

    outputDir = Path(FolderManager.instance().createTempFolder("qiime_output"))
    outputDataset = CustomDataset.createDataset(
        f"{experiment.id} - Step 3: Phylogenetic tree",
        experiment.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Workspace] Failed to create output dataset")

    for sample in denoisedSamples:
        index = sampleNumber(sample)
        processSample(index, sample, experiment, outputDataset, outputDir)


if __name__ == "__main__":
    initializeProject(main)
