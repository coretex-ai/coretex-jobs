import logging

from coretex import CustomDataset, Experiment
from coretex.project import initializeProject

from src.primer_trimming import primerTrimming
from src.multiplexed import demultiplexing
from src.demultiplexed import importDemultiplexedSamples
from src.denoise import denoise
from src.pylogenetic_analysis import phyogeneticDiversityAnalysis
from src.alpha_beta_diversity import alphaBetaDiversityAnalysis
from src.taxonomic_analysis import taxonomicAnalysis
from src.utils import isPairedEnd
from src.caching import getCache, uploadCacheAsArtifact, getCacheNameOne, getCacheNameTwo, getCacheNameThree, getCacheNameFour, getCacheNameFive


def main(experiment: Experiment[CustomDataset]):
    useCache = experiment.parameters["useCache"]
    experiment.dataset.download()

    pairedEnd = isPairedEnd(experiment.dataset)
    if pairedEnd:
        logging.info(">> [Microbiome analysis] Paired-end reads detected")
    else:
        logging.info(">> [Microbiome analysis] Single-end reads detected")

    if useCache:
        demultiplexedDataset = getCache(getCacheNameOne(experiment))
        if demultiplexedDataset is not None:
            demultiplexedDataset.download()
            uploadCacheAsArtifact(demultiplexedDataset, experiment)

    if not useCache or not demultiplexedDataset:
        #  Optional: Primer trimming
        if experiment.parameters["forwardAdapter"] or experiment.parameters["reverseAdapter"]:
            logging.info(">> [Microbiome analysis] Trimming primers based on provided adapter sequences with cutadapt")
            initialDataset = primerTrimming(experiment.dataset, experiment, pairedEnd)
            initialDataset.download()
        else:
            initialDataset = experiment.dataset

        # Step 1: Demultiplexing / Import to QIIME2
        logging.info(">> [Microbiome analysis] Step 1: Demux / Import")

        if experiment.parameters["barcodeColumn"]:
            demultiplexedDataset = demultiplexing(initialDataset, experiment)
        else:
            demultiplexedDataset = importDemultiplexedSamples(initialDataset, experiment, pairedEnd)

        demultiplexedDataset.download()

    # Step 2: Denoise
    if useCache:
        denoisedDataset = getCache(getCacheNameTwo(experiment))
        if denoisedDataset is not None:
            denoisedDataset.download()
            uploadCacheAsArtifact(denoisedDataset, experiment)

    if not useCache or not denoisedDataset:
        logging.info(">> [Microbiome analysis] Step 2: Denoise")
        denoisedDataset = denoise(demultiplexedDataset, experiment, pairedEnd)
        denoisedDataset.download()

    # Step 3: Phylogenetic Diversity Analysis
    if useCache:
        phylogeneticDataset = getCache(getCacheNameThree(experiment))
        if phylogeneticDataset is not None:
            phylogeneticDataset.download()
            uploadCacheAsArtifact(phylogeneticDataset, experiment)

    if not useCache or not phylogeneticDataset:
        logging.info(">> [Microbiome analysis] Step 3: Phylogenetic Diversity Analysis")
        phylogeneticDataset = phyogeneticDiversityAnalysis(denoisedDataset, experiment)
        phylogeneticDataset.download()

    # Step 4: Alpha and Beta Diversity Analysis
    if useCache:
        alphaBetaDataset = getCache(getCacheNameFour(experiment))
        if alphaBetaDataset is not None:
            alphaBetaDataset.download()
            uploadCacheAsArtifact(alphaBetaDataset, experiment)

    if not useCache or not alphaBetaDataset:
        logging.info(">> [Microbiome analysis] Step 4: Alpha and Beta Diversity Analysis")
        alphaBetaDiversityAnalysis(demultiplexedDataset, denoisedDataset, phylogeneticDataset, experiment)

    # Step 5: Taxonomic Analysis
    if useCache:
        taxonomicDataset = getCache(getCacheNameFive(experiment))
        if taxonomicDataset is not None:
            taxonomicDataset.download()
            uploadCacheAsArtifact(taxonomicDataset, experiment)

    if not useCache or not taxonomicDataset:
        logging.info(">> [Microbiome analysis] Step 5: Taxonomic Analysis")
        taxonomicAnalysis(demultiplexedDataset, denoisedDataset, experiment)


if __name__ == "__main__":
    initializeProject(main)
