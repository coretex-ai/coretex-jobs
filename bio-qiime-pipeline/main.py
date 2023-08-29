import logging

from coretex import SequenceDataset, Experiment, CustomDataset
from coretex.project import initializeProject
from coretex.bioinformatics import ctx_qiime2

from src.primer_trimming import primerTrimming
from src.import_multiplexed import importMultiplexed
from src.import_demultiplexed import importDemultiplexed
from src.demultiplexing import demultiplexing
from src.dada2 import denoise
from src.otu_clustering import otuClustering
from src.taxonomic_analysis import taxonomicAnalysis
from src.pylogenetic_analysis import phyogeneticDiversityAnalysis
from src.alpha_beta_diversity import alphaBetaDiversityAnalysis


def main(experiment: Experiment[CustomDataset]):
    initialDataset = experiment.dataset
    initialDataset.download()

    pairedEnd = None

    multiplexedFastqs = ctx_qiime2.getFastqMPSamples(initialDataset)
    if len(multiplexedFastqs) > 0:
        # Step 1: Import multiplexed sequences to QIIME2
        logging.info(">> [Microbiome analysis] Step 1: Import - Multiplexed")
        multiplexedDataset = importMultiplexed(multiplexedFastqs, experiment)
        multiplexedDataset.download()

        # Step 2: Demultiplexing
        logging.info(">> [Microbiome analysis] Step 2: Demultiplexing")
        demultiplexedDataset = demultiplexing(multiplexedDataset, experiment)
        demultiplexedDataset.download()
    else:
        sequenceDataset = SequenceDataset.fetchById(initialDataset.id)
        pairedEnd = sequenceDataset.isPairedEnd()

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
    logging.info(">> [Microbiome analysis] Step 2: Denoise")
    denoisedDataset = denoise(demultiplexedDataset, experiment, pairedEnd)
    denoisedDataset.download()

    # Step 4: OTU Clustering
    logging.info(">> [Microbiome analysis] Step: 4 OTU Clustering")
    otuClustering(denoisedDataset, experiment)

    # Step 5: Taxonomic Analysis
    logging.info(">> [Microbiome analysis] Step 5: Taxonomic Analysis")
    taxonomicAnalysis(demultiplexedDataset, denoisedDataset, experiment)

    # Step 6: Phylogenetic Diversity Analysis
    logging.info(">> [Microbiome analysis] Step 6: Phylogenetic Diversity Analysis")
    phylogeneticDataset = phyogeneticDiversityAnalysis(denoisedDataset, experiment)
    phylogeneticDataset.download()

    # Step 7: Alpha and Beta Diversity Analysis
    logging.info(">> [Microbiome analysis] Step 7: Alpha and Beta Diversity Analysis")
    alphaBetaDiversityAnalysis(demultiplexedDataset, denoisedDataset, phylogeneticDataset, experiment)


if __name__ == "__main__":
    initializeProject(main)
