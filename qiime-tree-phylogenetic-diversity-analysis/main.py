from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomDataset, CustomSample, TaskRun, folder_manager, currentTaskRun
from coretex.bioinformatics import ctx_qiime2


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
    taskRun: TaskRun,
    outputDataset: CustomDataset,
    outputDir: Path
) -> None:

    sample.unzip()

    sampleOutputDir = outputDir / str(sample.id)
    sampleOutputDir.mkdir()

    # Phylogenetic diversity analysis
    logging.info(">> [Qiime: Phylogenetic Diversity] Generating phylogenetic tree")
    treePath = phylogenyAlignToTreeMafftFasttreeSample(sample, sampleOutputDir)
    ctx_qiime2.createSample(f"{index}-phylogenetic-tree", outputDataset.id, treePath, taskRun, "Step 6: Phylogenetic tree")


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    taskRun.dataset.download()

    denoisedSamples = ctx_qiime2.getDenoisedSamples(taskRun.dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Qiime: Phylogenetic Diversity] Dataset has 0 denoised samples")

    outputDir = folder_manager.createTempFolder("tree_output")
    outputDataset = CustomDataset.createDataset(
        f"{taskRun.id} - Step 6: Phylogenetic tree",
        taskRun.spaceId
    )

    if outputDataset is None:
        raise ValueError(">> [Qiime: Phylogenetic Diversity] Failed to create output dataset")

    for sample in denoisedSamples:
        index = ctx_qiime2.sampleNumber(sample)
        processSample(index, sample, taskRun, outputDataset, outputDir)


if __name__ == "__main__":
    main()
