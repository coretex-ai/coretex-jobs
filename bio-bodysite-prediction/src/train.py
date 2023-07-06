from pathlib import Path

import logging
import pickle
import time

from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import sparse

import numpy as np
import xgboost as xgb

from coretex import Experiment, CustomDataset, ExperimentStatus
from coretex.folder_management import FolderManager

from .utils import savePredictionFile


class Progress(TrainingCallback):

    """
    Callback function for XGBoost showing the loss and accuracy of the model at the current epoch.\n
    The metrics will be printed every n rounds based on the value of updateRound.
    """

    def __init__(
        self, updateRound: int,
        evalSet: list[tuple[sparse.csr_matrix, np.ndarray]],
        experiment: Experiment
    ) -> None:

        self.updateRound = updateRound
        self.experiment = experiment
        self.eval = xgb.DMatrix(evalSet[0][0], label = evalSet[0][1])
        self.yEval = evalSet[0][1]

    def after_iteration(self, model: XGBClassifier, epoch: int, evals_log) -> bool:
        for data, metric in evals_log.items():
            for metricName, log in metric.items():
                loss = log[-1]
                yPred = model.predict(self.eval)

                if len(yPred.shape) > 1:
                    acc = accuracy_score(self.yEval, np.argmax(yPred, axis = 1))
                else:
                    # This happens if there is only 1 class to predict
                    acc = 1 - np.average(np.absolute(self.yEval - yPred))

                self.experiment.submitMetrics({
                    "loss": (epoch + 1, loss),
                    "accuracy": (epoch + 1, acc)
                })

                if (epoch + 1) % self.updateRound == 0:
                    logging.info(f">> [XGBoost] Round: {epoch + 1:>3d}, Loss: {loss:>7f}, Accuracy: {acc *100:>0.2f}%")

        return False


def train(
    experiment: Experiment[CustomDataset],
    inputTable: sparse.csr_matrix,
    outputTable: np.ndarray,
    uniqueBodySites: dict[str, int],
    uniqueTaxons: dict[str, int],
    sampleIdList: list
) -> float:

    savePredictionFilePath = Path(FolderManager.instance().temp) / "body_site_predictions.csv"
    modelPath = Path(FolderManager.instance().getTempFolder("modelFolder"))

    experiment.updateStatus(ExperimentStatus.inProgress, "Training XGBoost model")

    learningRate = experiment.parameters["learningRate"]
    epochs = experiment.parameters["epochs"]
    earlyStopping = experiment.parameters["earlyStopping"]
    validationSplit = experiment.parameters["validationSplit"]
    useGpu = experiment.parameters["useGpu"]

    xTrain, xTest, yTrain, yTest = train_test_split(
        inputTable,
        outputTable,
        test_size = validationSplit,
        random_state = 1,
        stratify = outputTable
    )

    logging.info(">> [MicrobiomeForensics] Instantiating XGBClassifier and starting training")

    evalSet = [(xTest, yTest)]
    progress = Progress(10, evalSet, experiment)

    treeMethod = "gpu_hist" if useGpu else "exact"

    model = XGBClassifier(
        eval_metric = 'mlogloss',
        callbacks = [progress],
        early_stopping_rounds = earlyStopping,
        n_estimators = epochs,
        learning_rate = learningRate,
        tree_method = treeMethod
    )

    start = time.time()
    model.fit(xTrain, yTrain, eval_set = evalSet, verbose = False)
    trainTime = time.time() - start

    yPred = model.predict(xTest)
    zPred = model.predict(xTrain)

    accuracy = accuracy_score(yTest, yPred)
    logging.info(f">> [MicrobiomeForensics] Training finished in {trainTime:>0.1f}s with accuracy: {accuracy * 100:>0.2f}%")

    experiment.updateStatus(ExperimentStatus.inProgress, "Saving model and associated data")

    savePredictionFile(
        experiment,
        savePredictionFilePath,
        xTrain,
        xTest,
        sampleIdList,
        uniqueBodySites,
        yTrain,
        yTest,
        yPred,
        zPred
    )

    model.save_model(modelPath / f"model.txt")

    with open(modelPath / "uniqueTaxons.pkl", "wb") as f:
        pickle.dump(uniqueTaxons, f)

    with open(modelPath / "uniqueBodySites.pkl", "wb") as f:
        pickle.dump(uniqueBodySites, f)

    return accuracy
