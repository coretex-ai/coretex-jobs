from typing import Any, Optional, Union
from pathlib import Path

import logging
import pickle

from coretex import TaskRun, CustomDataset, Model, folder_manager, currentTaskRun
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas import DataFrame

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def loadData(dataset: CustomDataset, metadataFileName: str, targetMetadataColumn: str) -> tuple[DataFrame, DataFrame]:
    dataset.download()
    if dataset.count != 1:
        raise ValueError(f">> [Body Site Prediction] Expected to find 1 sample in the dataset. Found {dataset.count}")

    sample = dataset.samples[0]
    sample.unzip()

    sampleContent = list(sample.path.glob("*.csv"))

    fileCount = len(sampleContent)
    if fileCount != 2:
        raise ValueError(f">> [Body Site Prediction] Expected to find 2 csv files in the sample (OTU table and metadata files). Found {fileCount}")

    metadataPath = sample.path / metadataFileName
    if not metadataPath.exists():
        raise ValueError(f">> [Body Site Prediction] Could not find metadata file ({metadataFileName}) in the sample")

    metadataTable = pd.read_csv(metadataPath)
    bodySites = metadataTable[targetMetadataColumn]

    for filePath in sampleContent:
        if filePath.name == metadataFileName:
            continue

        otuTable = pd.read_csv(filePath)

    return otuTable, bodySites


def train(
    xTrain: DataFrame,
    yTrain: DataFrame,
    criterion: str,
    nEstimators: int,
    maxFeatures: Optional[Any],
    maxDepth: Optional[int],
    bootstrap: bool,
    minSamplesSplit: Union[int, float],
    minWeightFractionLeaf: float,
    minSamplesLeaf: Union[int, float],
    seed: int = 42
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_jobs = -1,
        class_weight = "balanced",
        random_state = seed,
        criterion = criterion,
        n_estimators = nEstimators,
        max_features = maxFeatures,
        max_depth = maxDepth,
        bootstrap = bootstrap,
        min_samples_split = minSamplesSplit,
        min_weight_fraction_leaf = minWeightFractionLeaf,
        min_samples_leaf = minSamplesLeaf
    )

    model.fit(xTrain, yTrain)

    return model


def generateConfusionMatrix(
    plotTitle: GridSearchCV,
    groudTruth: DataFrame,
    predicted: DataFrame,
    plotPath: Path,
    taskRun: TaskRun
) -> None:

    logging.info(">> [Body Site Prediction] Generating confusion matrix")

    cm = confusion_matrix(groudTruth, predicted, labels = np.unique(predicted))
    sns.heatmap(
        cm,
        xticklabels = np.unique(predicted),
        yticklabels = np.unique(predicted),
        linewidths = 3,
        linecolor = "white",
        cmap = sns.cubehelix_palette(start = 2, as_cmap = True),
        annot = True
    )

    plt.title(plotTitle)
    plt.savefig(plotPath, bbox_inches = "tight", dpi = 300)

    if taskRun.createArtifact(plotPath, plotPath.name) is None:
        logging.error(f">> [Body Site Prediction] Failed to create {plotPath.name}")


def createProbabilityTable(model: GridSearchCV, x: DataFrame, sampleIds: pd.Series, taskRun: TaskRun) -> None:
    logging.info(">> [Body Site Prediction] Creating probability table")

    probabilityTable = model.predict_proba(x)

    probs = pd.DataFrame(probabilityTable, columns = model.classes_, index = sampleIds)
    savePath = folder_manager.temp / "probabilities.csv"
    probs.to_csv(savePath, index=True)
    if taskRun.createArtifact(savePath, savePath.name) is None:
        logging.info(f">> [Body Site Prediction] Failed to create {savePath.name}")


def saveFeatureWeights(model: RandomForestClassifier, xTrain: DataFrame, taskRun: TaskRun) -> None:
    logging.info(">> [Body Site Prediction] Saving feature weights")

    savePath = folder_manager.temp / "feature_importances.csv"

    featureW = pd.DataFrame({
        "feature": xTrain.columns,
        "weight": model.feature_importances_
    })

    featureWsorted = featureW.sort_values(
        by = ["weight"],
        ascending = False,
        key = lambda col: col.abs()
    )

    featureWsorted.loc[featureWsorted["weight"] != 0]

    featureW.to_csv(savePath, index = False)

    if taskRun.createArtifact(savePath, savePath.name) is None:
        logging.error(f">> [Body Site Prediction] Failed to create {savePath.name}")


def generateClassificationReport(groundTruth: DataFrame, prediction: np.ndarray, taskRun: TaskRun) -> None:
    logging.info(">> [Body Site Prediction] Generating classification report")

    report = classification_report(groundTruth, prediction, output_dict = True)

    savePath = folder_manager.temp / "classfication_report.csv"
    DataFrame(report).to_csv(savePath)

    if taskRun.createArtifact(savePath, savePath.name) is None:
        logging.error(f">> [Body Site Prediction] Failed to create {savePath.name}")


def saveModel(model: GridSearchCV, name: str, accuracy: float, taskRun: TaskRun) -> Model:
    logging.info(">> [Body Site Prediction] Saving model")

    modelPath = folder_manager.createTempFolder("model")

    savePath = modelPath / f"{name}.pkl"
    with open(savePath, "wb") as file:
        pickle.dump(model, file)

    outputModel = Model.createModel(f"{taskRun.id}-random-forest-classifier", taskRun.id, accuracy, {})
    outputModel.upload(modelPath)

    return outputModel


def main() -> None:
    taskRun = currentTaskRun()

    logging.info(">> [Body Site Prediction] Loading data")
    otuTable, bodySites = loadData(
        taskRun.dataset,
        taskRun.parameters["metadataFileName"],
        taskRun.parameters["targetMetadataColumn"]
    )

    xTrain, xTest, yTrain, yTest = train_test_split(
        otuTable,
        bodySites,
        random_state = 1,
        stratify = bodySites,
        test_size = taskRun.parameters["validationSplit"]
    )

    sampleIdTest = xTest["sampleid"]

    xTrain = pd.get_dummies(xTrain.drop(columns="sampleid"), drop_first=True)
    xTest = pd.get_dummies(xTest.drop(columns="sampleid"), drop_first=True)

    logging.info(">> [Body Site Prediction] Starting training")
    model = train(
        xTrain,
        yTrain,
        taskRun.parameters["criterion"],
        taskRun.parameters["nEstimators"],
        taskRun.parameters["maxFeatures"],
        taskRun.parameters["maxDepth"],
        taskRun.parameters["bootstrap"],
        taskRun.parameters["minSamplesSplit"],
        taskRun.parameters["minWeightFractionLeaf"],
        taskRun.parameters["minSamplesLeaf"],
        taskRun.parameters["seed"]
    )

    yPredTest = model.predict(xTest)
    accuracy = accuracy_score(yTest, yPredTest)

    logging.info(f">> [Body Site Prediction] Accuracy: {accuracy}")

    conMatrixTitle = "RF training accuracy: " + str(accuracy)
    conMatrixPath = folder_manager.temp / "confusion_matrix.png"
    generateConfusionMatrix(conMatrixTitle, yTest, yPredTest, conMatrixPath, taskRun)

    createProbabilityTable(model, xTest, sampleIdTest, taskRun)
    saveFeatureWeights(model, xTrain, taskRun)
    generateClassificationReport(yTest, yPredTest, taskRun)

    outputModel = saveModel(model, "classifier", accuracy, taskRun)
    taskRun.submitOutput("outputModel", outputModel)


if __name__ == "__main__":
    main()
