import logging

from coretex import SequenceDataset, Run, CustomDataset
from coretex.job import initializeJob

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


def main(run: Run[CustomDataset]):
    useCache = run.parameters["useCache"]
    initialDataset = run.dataset
    initialDataset.download()

    pairedEnd = None

    cacheNameOne = getCacheNameOne(run)
    if run.parameters["barcodeColumn"]:
        if useCache and cacheExists(cacheNameOne):
            multiplexedDataset = getCache(cacheNameOne, run)
        else:
            # Step 1: Import multiplexed sequences to QIIME2
            logging.info(">> [Microbiome analysis] Step 1: Import - Multiplexed")
            multiplexedDataset = importMultiplexed(initialDataset, run)
            multiplexedDataset.download()

        cacheNameTwo = getCacheNameTwo(run)
        if useCache and cacheExists(cacheNameTwo):
            demultiplexedDataset = getCache(cacheNameTwo, run)
        else:
            # Step 2: Demultiplexing
            logging.info(">> [Microbiome analysis] Step 2: Demultiplexing")
            demultiplexedDataset = demultiplexing(multiplexedDataset, run)
            demultiplexedDataset.download()
    else:
        sequenceDataset = SequenceDataset.decode(initialDataset.encode())
        pairedEnd = sequenceDataset.isPairedEnd()

        if useCache and cacheExists(cacheNameOne):
            demultiplexedDataset = getCache(cacheNameOne, run)
        else:
            # Primer Trimming with Cutadapt
            if run.parameters["forwardAdapter"] or run.parameters["reverseAdapter"]:
                logging.info(">> [Microbiome analysis] Trimming primers based on provided adapter sequences with cutadapt")
                sequenceDataset = primerTrimming(sequenceDataset, run, pairedEnd)
                sequenceDataset.download()

            # Step 1: Import demultiplexed sequences to QIIME2
            logging.info(">> [Microbiome analysis] Step 1: Import - Demultiplexed")
            demultiplexedDataset = importDemultiplexed(
                sequenceDataset,
                run,
                pairedEnd
            )
            demultiplexedDataset.download()

        logging.info(">> [Microbiome analysis] Skipping \"Step 2: Demultiplexing\" because data is already demultiplexed")

    # Step 3: DADA2
    cacheNameThree = getCacheNameThree(run)
    if useCache and cacheExists(cacheNameThree):
        denoisedDataset = getCache(cacheNameThree, run)
    else:
        logging.info(">> [Microbiome analysis] Step 2: Denoise")
        denoisedDataset = denoise(demultiplexedDataset, run, pairedEnd)
        denoisedDataset.download()

    # Step 4: OTU Clustering
    cacheNameFour = getCacheNameFour(run)
    if useCache and cacheExists(cacheNameFour):
        clusteredDataset = getCache(cacheNameFour, run)
    else:
        logging.info(">> [Microbiome analysis] Step: 4 OTU Clustering")
        otuClustering(denoisedDataset, run)

    # Step 5: Taxonomic Analysis
    cacheNameFive = getCacheNameFive(run)
    if useCache and cacheExists(cacheNameFive):
        taxonomicDataset = getCache(cacheNameFive, run)
    else:
        logging.info(">> [Microbiome analysis] Step 5: Taxonomic Analysis")
        taxonomicAnalysis(demultiplexedDataset, denoisedDataset, run)

    # Step 6: Phylogenetic Diversity Analysis
    cacheNameSix = getCacheNameSix(run)
    if useCache and cacheExists(cacheNameSix):
        phylogeneticDataset = getCache(cacheNameSix, run)
    else:
        logging.info(">> [Microbiome analysis] Step 6: Phylogenetic Diversity Analysis")
        phylogeneticDataset = phyogeneticDiversityAnalysis(denoisedDataset, run)
        phylogeneticDataset.download()

    # Step 7: Alpha and Beta Diversity Analysis
    cacheNameSeven = getCacheNameSeven(run)
    if useCache and cacheExists(cacheNameSeven):
        alphaBetaDataset = getCache(cacheNameSeven, run)
    else:
        logging.info(">> [Microbiome analysis] Step 7: Alpha and Beta Diversity Analysis")
        alphaBetaDiversityAnalysis(demultiplexedDataset, denoisedDataset, phylogeneticDataset, run)


if __name__ == "__main__":
    initializeJob(main)
