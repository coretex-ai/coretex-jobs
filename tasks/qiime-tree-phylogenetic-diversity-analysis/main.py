from typing import Optional
from pathlib import Path
from zipfile import ZipFile

import logging

from coretex import CustomDataset, CustomSample, TaskRun, folder_manager, currentTaskRun, createDataset
from coretex.bioinformatics import ctx_qiime2


def phylogenyAlignToTreeMafftFasttreeSample(sample: CustomSample, outputDir: Path, threads: Optional[int]) -> Path:
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
        str(rootedTreePath),
        threads
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
    treePath = phylogenyAlignToTreeMafftFasttreeSample(sample, sampleOutputDir, taskRun.parameters["threads"])
    ctx_qiime2.createSample(f"{index}-phylogenetic-tree", outputDataset, treePath, taskRun, "Step 6: Phylogenetic tree")


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    taskRun.dataset.download()

    denoisedSamples = ctx_qiime2.getDenoisedSamples(taskRun.dataset)
    if len(denoisedSamples) == 0:
        raise ValueError(">> [Qiime: Phylogenetic Diversity] Dataset has 0 denoised samples")

    outputDir = folder_manager.createTempFolder("tree_output")

    outputDatasetName = f"{taskRun.id} - Step 6: Phylogenetic tree"
    with createDataset(CustomDataset, outputDatasetName, taskRun.projectId) as outputDataset:
        for sample in denoisedSamples:
            index = ctx_qiime2.sampleNumber(sample)
            processSample(index, sample, taskRun, outputDataset, outputDir)

    taskRun.submitOutput("outputDataset", outputDataset)


if __name__ == "__main__":
    main()
