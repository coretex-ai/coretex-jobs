import logging

from coretex import SequenceDataset, TaskRun, CustomDataset, currentTaskRun

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


def main() -> None:
    taskRun: TaskRun[CustomDataset] = currentTaskRun()

    useCache = taskRun.parameters["useCache"]
    initialDataset = taskRun.dataset
    initialDataset.download()

    pairedEnd = None

    cacheNameOne = getCacheNameOne(taskRun)
    if taskRun.parameters["barcodeColumn"]:
        if useCache and cacheExists(cacheNameOne):
            multiplexedDataset = getCache(cacheNameOne, taskRun)
        else:
            # Step 1: Import multiplexed sequences to QIIME2
            logging.info(">> [Microbiome analysis] Step 1: Import - Multiplexed")
            multiplexedDataset = importMultiplexed(initialDataset, taskRun)
            multiplexedDataset.download()

        cacheNameTwo = getCacheNameTwo(taskRun)
        if useCache and cacheExists(cacheNameTwo):
            demultiplexedDataset = getCache(cacheNameTwo, taskRun)
        else:
            # Step 2: Demultiplexing
            logging.info(">> [Microbiome analysis] Step 2: Demultiplexing")
            demultiplexedDataset = demultiplexing(multiplexedDataset, taskRun)
            demultiplexedDataset.download()
    else:
        sequenceDataset = SequenceDataset.decode(initialDataset.encode())
        pairedEnd = sequenceDataset.isPairedEnd()

        if useCache and cacheExists(cacheNameOne):
            demultiplexedDataset = getCache(cacheNameOne, taskRun)
        else:
            # Primer Trimming with Cutadapt
            if taskRun.parameters["forwardAdapter"] or taskRun.parameters["reverseAdapter"]:
                logging.info(">> [Microbiome analysis] Trimming primers based on provided adapter sequences with cutadapt")
                sequenceDataset = primerTrimming(sequenceDataset, taskRun, pairedEnd)
                sequenceDataset.download()

            # Step 1: Import demultiplexed sequences to QIIME2
            logging.info(">> [Microbiome analysis] Step 1: Import - Demultiplexed")
            demultiplexedDataset = importDemultiplexed(
                sequenceDataset,
                taskRun,
                pairedEnd
            )
            demultiplexedDataset.download()

        logging.info(">> [Microbiome analysis] Skipping \"Step 2: Demultiplexing\" because data is already demultiplexed")

    # Step 3: DADA2
    cacheNameThree = getCacheNameThree(taskRun)
    if useCache and cacheExists(cacheNameThree):
        denoisedDataset = getCache(cacheNameThree, taskRun)
    else:
        logging.info(">> [Microbiome analysis] Step 3: DADA2")
        denoisedDataset = denoise(demultiplexedDataset, taskRun, pairedEnd)
        denoisedDataset.download()

    # Step 4: OTU Clustering
    cacheNameFour = getCacheNameFour(taskRun)
    if useCache and cacheExists(cacheNameFour):
        clusteredDataset = getCache(cacheNameFour, taskRun)
    else:
        logging.info(">> [Microbiome analysis] Step: 4 OTU Clustering")
        otuClustering(denoisedDataset, taskRun)

    # Step 5: Taxonomic Analysis
    cacheNameFive = getCacheNameFive(taskRun)
    if useCache and cacheExists(cacheNameFive):
        taxonomicDataset = getCache(cacheNameFive, taskRun)
    else:
        logging.info(">> [Microbiome analysis] Step 5: Taxonomic Analysis")
        taxonomicAnalysis(demultiplexedDataset, denoisedDataset, taskRun)

    # Step 6: Phylogenetic Diversity Analysis
    cacheNameSix = getCacheNameSix(taskRun)
    if useCache and cacheExists(cacheNameSix):
        phylogeneticDataset = getCache(cacheNameSix, taskRun)
    else:
        logging.info(">> [Microbiome analysis] Step 6: Phylogenetic Diversity Analysis")
        phylogeneticDataset = phyogeneticDiversityAnalysis(denoisedDataset, taskRun)
        phylogeneticDataset.download()

    # Step 7: Alpha and Beta Diversity Analysis
    cacheNameSeven = getCacheNameSeven(taskRun)
    if useCache and cacheExists(cacheNameSeven):
        alphaBetaDataset = getCache(cacheNameSeven, taskRun)
    else:
        logging.info(">> [Microbiome analysis] Step 7: Alpha and Beta Diversity Analysis")
        diversityAnalysisDataset = alphaBetaDiversityAnalysis(demultiplexedDataset, denoisedDataset, phylogeneticDataset, taskRun)

    taskRun.submitOutput("outputDataset", diversityAnalysisDataset)


if __name__ == "__main__":
    main()
