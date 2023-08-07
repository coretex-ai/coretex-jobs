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


def main(experiment: Experiment[CustomDataset]):
    experiment.dataset.download()

    pairedEnd = isPairedEnd(experiment.dataset, experiment)
    if pairedEnd:
        logging.info(">> [Microbiome analysis] Paired-end reads detected")
    else:
        logging.info(">> [Microbiome analysis] Single-end reads detected")

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
        step1dataset = demultiplexing(initialDataset, experiment)
    else:
        step1dataset = importDemultiplexedSamples(initialDataset, experiment, pairedEnd)

    # Step 2: Denoise
    logging.info(">> [Microbiome analysis] Step 2: Denoise")
    step1dataset.download()
    step2dataset = denoise(step1dataset, experiment, pairedEnd)

    # Step 3: Phylogenetic Diversity Analysis
    logging.info(">> [Microbiome analysis] Step 3: Phylogenetic Diversity Analysis")
    step2dataset.download()
    step3dataset = phyogeneticDiversityAnalysis(step2dataset, experiment)

    # Step 4: Alpha and Beta Diversity Analysis
    logging.info(">> [Microbiome analysis] Step 4: Alpha and Beta Diversity Analysis")
    step3dataset.download()
    alphaBetaDiversityAnalysis(step1dataset, step2dataset, step3dataset, experiment)

    # Step 5: Taxonomic Analysis
    logging.info(">> [Microbiome analysis] Step 5: Taxonomic Analysis")

    taxonomicAnalysis(step1dataset, step2dataset, experiment)


if __name__ == "__main__":
    initializeProject(main)
