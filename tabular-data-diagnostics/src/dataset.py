import glob
import os
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame

import pandas as pd

from coretex import CustomDataset, Experiment

from .data_type import DataType


def _loadSamples(dataset: CustomDataset) -> DataFrame:
    dataList: list[DataFrame] = []

    for sample in dataset.samples:
        sample.unzip()

        for dataType in DataType:
            dataPath = glob.glob(os.path.join(sample.path, f"*.{dataType.extension}"))

            for f in dataPath:
                if not os.path.exists(f):
                    continue

                df = pd.read_csv(f, delimiter = dataType.delimiter)
                dataList.append(df)

    logging.info(f">> [Tabular Data Diagnostics] Dataset loaded successfully, id: {dataset.id}.")

    return pd.concat(dataList)


def loadDataset(coretexDataset: CustomDataset, validationSplit: float, labelColumn: str, excludeColumns: list[str]) -> tuple[DataFrame, DataFrame, list[str]]:
    logging.info(">> [Tabular Data Diagnostics] Downloading dataset...")

    coretexDataset.download()

    logging.info(">> [Tabular Data Diagnostics] Loading dataset...")

    mergedDf = _loadSamples(coretexDataset)

    if excludeColumns is not None:
        logging.info(f">> [Tabular Data Diagnostics] Excluding columns: {excludeColumns}.")
        for column in excludeColumns:
            mergedDf.drop(column, inplace=True, axis=1)

    logging.info(f">> [Tabular Data Diagnostics] Encoding labelColumn - '{labelColumn}'.")
    mergedDf[labelColumn] = LabelEncoder().fit_transform(mergedDf[labelColumn])
    mergedDf = mergedDf.dropna(axis = 1, how = "all")

    filteredColumns = mergedDf.dtypes[mergedDf.dtypes == object]
    listOfColumnNames = list(filteredColumns.index)

    for column in listOfColumnNames:
        mergedDf[column] = LabelEncoder().fit_transform(mergedDf[column])

    logging.info(f">> [Tabular Data Diagnostics] Splitting data on train and test.")
    train, test = train_test_split(mergedDf, test_size = validationSplit, random_state = 0)

    labelsRaw = mergedDf[labelColumn].unique().tolist()
    labels = [str(label) for label in labelsRaw]

    return train, test, labels


def extractXY(df: DataFrame, labelColumn: str) -> tuple[DataFrame, DataFrame]:
    x = df.drop(labelColumn, axis = 1)
    y = df[labelColumn]

    return x, y


def extractTestTrainData(train: DataFrame, test: DataFrame, labelColumn: str) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    logging.info(f">> [Tabular Data Diagnostics] Extracting test and train data...")
    xTrain, yTrain = extractXY(train, labelColumn)
    xTest, yTest = extractXY(test, labelColumn)

    scX = StandardScaler()
    xTrain = scX.fit_transform(xTrain)
    xTest = scX.fit_transform(xTest)

    return xTrain, xTest, yTrain, yTest
