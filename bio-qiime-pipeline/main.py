import logging

from coretex import SequenceDataset, Experiment, CustomDataset
from coretex.project import initializeProject

from src.primer_trimming import primerTrimming
from src.multiplexed import demultiplexing
from src.demultiplexed import importDemultiplexedSamples
from src.denoise import denoise
from src.pylogenetic_analysis import phyogeneticDiversityAnalysis
from src.alpha_beta_diversity import alphaBetaDiversityAnalysis
from src.taxonomic_analysis import taxonomicAnalysis
from src.caching import getCache, cacheExists, getCacheNameOne, getCacheNameTwo, getCacheNameThree, getCacheNameFour, getCacheNameFive


def main(experiment: Experiment[SequenceDataset]):
    useCache = experiment.parameters["useCache"]
    experiment.dataset.download()

    pairedEnd = experiment.dataset.isPairedEnd()
    if pairedEnd:
        logging.info(">> [Microbiome analysis] Paired-end reads detected")
    else:
        logging.info(">> [Microbiome analysis] Single-end reads detected")

    cacheNameOne = getCacheNameOne(experiment)
    if useCache and cacheExists(cacheNameOne):
        demultiplexedDataset = getCache(cacheNameOne, experiment)
    else:
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
            demultiplexedDataset = demultiplexing(CustomDataset.fetchById(initialDataset.id), experiment)
        else:
            demultiplexedDataset = importDemultiplexedSamples(initialDataset, experiment, pairedEnd)

        demultiplexedDataset.download()

    # Step 2: Denoise
    cacheNameTwo = getCacheNameTwo(experiment)
    if useCache and cacheExists(cacheNameTwo):
        denoisedDataset = getCache(cacheNameTwo, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step 2: Denoise")
        denoisedDataset = denoise(demultiplexedDataset, experiment, pairedEnd)
        denoisedDataset.download()

    # Step 3: Phylogenetic Diversity Analysis
    cacheNameThree = getCacheNameThree(experiment)
    if useCache and cacheExists(cacheNameThree):
        phylogeneticDataset = getCache(cacheNameThree, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step 3: Phylogenetic Diversity Analysis")
        phylogeneticDataset = phyogeneticDiversityAnalysis(denoisedDataset, experiment)
        phylogeneticDataset.download()

    # Step 4: Alpha and Beta Diversity Analysis
    cacheNameFour = getCacheNameFour(experiment)
    if useCache and cacheExists(cacheNameFour):
        alphaBetaDataset = getCache(cacheNameFour, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step 4: Alpha and Beta Diversity Analysis")
        alphaBetaDiversityAnalysis(demultiplexedDataset, denoisedDataset, phylogeneticDataset, experiment)

    # Step 5: Taxonomic Analysis
    cacheNameFive = getCacheNameFive(experiment)
    if useCache and cacheExists(cacheNameFive):
        taxonomicDataset = getCache(cacheNameFive, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step 5: Taxonomic Analysis")
        taxonomicAnalysis(demultiplexedDataset, denoisedDataset, experiment)


if __name__ == "__main__":
    initializeProject(main, SequenceDataset)
