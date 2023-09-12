import logging

from coretex import SequenceDataset, Experiment, CustomDataset, currentExperiment

from src.primer_trimming import primerTrimming
from src.import_multiplexed import importMultiplexed
from src.import_demultiplexed import importDemultiplexed
from src.demultiplexing import demultiplexing
from src.dada2 import denoise
from src.otu_clustering import otuClustering
from src.taxonomic_analysis import taxonomicAnalysis
from src.pylogenetic_analysis import phyogeneticDiversityAnalysis
from src.alpha_beta_diversity import alphaBetaDiversityAnalysis
from src.caching import getCache, cacheExists, getCacheNameOne, getCacheNameTwo, getCacheNameThree, getCacheNameFour, getCacheNameFive, getCacheNameSix, getCacheNameSeven


def main():
    experiment: Experiment[CustomDataset] = currentExperiment()

    useCache = experiment.parameters["useCache"]
    initialDataset = experiment.dataset
    initialDataset.download()

    pairedEnd = None

    cacheNameOne = getCacheNameOne(experiment)
    if experiment.parameters["barcodeColumn"]:
        if useCache and cacheExists(cacheNameOne):
            multiplexedDataset = getCache(cacheNameOne, experiment)
        else:
            # Step 1: Import multiplexed sequences to QIIME2
            logging.info(">> [Microbiome analysis] Step 1: Import - Multiplexed")
            multiplexedDataset = importMultiplexed(initialDataset, experiment)
            multiplexedDataset.download()

        cacheNameTwo = getCacheNameTwo(experiment)
        if useCache and cacheExists(cacheNameTwo):
            demultiplexedDataset = getCache(cacheNameTwo, experiment)
        else:
            # Step 2: Demultiplexing
            logging.info(">> [Microbiome analysis] Step 2: Demultiplexing")
            demultiplexedDataset = demultiplexing(multiplexedDataset, experiment)
            demultiplexedDataset.download()
    else:
        sequenceDataset = SequenceDataset.decode(initialDataset.encode())
        pairedEnd = sequenceDataset.isPairedEnd()

        if useCache and cacheExists(cacheNameOne):
            demultiplexedDataset = getCache(cacheNameOne, experiment)
        else:
            # Primer Trimming with Cutadapt
            if experiment.parameters["forwardAdapter"] or experiment.parameters["reverseAdapter"]:
                logging.info(">> [Microbiome analysis] Trimming primers based on provided adapter sequences with cutadapt")
                sequenceDataset = primerTrimming(sequenceDataset, experiment, pairedEnd)
                sequenceDataset.download()

            # Step 1: Import demultiplexed sequences to QIIME2
            logging.info(">> [Microbiome analysis] Step 1: Import - Demultiplexed")
            demultiplexedDataset = importDemultiplexed(
                sequenceDataset,
                experiment,
                pairedEnd
            )
            demultiplexedDataset.download()

        logging.info(">> [Microbiome analysis] Skipping \"Step 2: Demultiplexing\" because data is already demultiplexed")

    # Step 3: DADA2
    cacheNameThree = getCacheNameThree(experiment)
    if useCache and cacheExists(cacheNameThree):
        denoisedDataset = getCache(cacheNameThree, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step 3: DADA2")
        denoisedDataset = denoise(demultiplexedDataset, experiment, pairedEnd)
        denoisedDataset.download()

    # Step 4: OTU Clustering
    cacheNameFour = getCacheNameFour(experiment)
    if useCache and cacheExists(cacheNameFour):
        clusteredDataset = getCache(cacheNameFour, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step: 4 OTU Clustering")
        otuClustering(denoisedDataset, experiment)

    # Step 5: Taxonomic Analysis
    cacheNameFive = getCacheNameFive(experiment)
    if useCache and cacheExists(cacheNameFive):
        taxonomicDataset = getCache(cacheNameFive, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step 5: Taxonomic Analysis")
        taxonomicAnalysis(demultiplexedDataset, denoisedDataset, experiment)

    # Step 6: Phylogenetic Diversity Analysis
    cacheNameSix = getCacheNameSix(experiment)
    if useCache and cacheExists(cacheNameSix):
        phylogeneticDataset = getCache(cacheNameSix, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step 6: Phylogenetic Diversity Analysis")
        phylogeneticDataset = phyogeneticDiversityAnalysis(denoisedDataset, experiment)
        phylogeneticDataset.download()

    # Step 7: Alpha and Beta Diversity Analysis
    cacheNameSeven = getCacheNameSeven(experiment)
    if useCache and cacheExists(cacheNameSeven):
        alphaBetaDataset = getCache(cacheNameSeven, experiment)
    else:
        logging.info(">> [Microbiome analysis] Step 7: Alpha and Beta Diversity Analysis")
        alphaBetaDiversityAnalysis(demultiplexedDataset, denoisedDataset, phylogeneticDataset, experiment)


if __name__ == "__main__":
    main()
