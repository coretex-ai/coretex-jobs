import logging
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from coretex import CustomDataset, Run, Model, folder_manager
from coretex.job import initializeJob

from src.dataset import extractTestTrainData, loadDataset


def saveModel(run: Run[CustomDataset], accuracy: float, trainColumnCount: int, labels: list[str]):
    model = Model.createModel(run.name, run.id, accuracy, {})
    modelPath = folder_manager.temp / "model"

    model.saveModelDescriptor(modelPath, {
        "project_task": run.spaceTask,
        "labels": labels,
        "modelName": model.name,
        "description": run.description,

        "input_description": """
            Input shape is [numberOfSamples, columnValues]

            Number of samples - the row count of the tabular data used as the input
            Column values - the value of the columns sorted in the order that they were used
                            during the training process
        """,
        "input_shape": [None, trainColumnCount],

        "output_description": """
            Output shape - [numberOfSamples, predictedClass]

            Number of samples - the prediction for every row that was sent as the input to the model
            Predicted class - index of the predicted class
        """,
        "output_shape": [None, len(labels)]
    })

    model.upload(modelPath)


def main(run: Run[CustomDataset]):
    modelPath = folder_manager.createTempFolder("model")

    train, test, labels = loadDataset(
        run.dataset,
        run.parameters["validationSplit"],
        run.parameters["labelColumn"],
        run.parameters["excludeColumns"]
    )

    xTrain, xTest, yTrain, yTest = extractTestTrainData(
        train,
        test,
        run.parameters["labelColumn"]
    )

    logging.info(">> [Tabular Data Diagnostics] Starting with training the model...")

    classifier = RandomForestClassifier(
        n_estimators = int(run.parameters["nEstimators"]),
        max_depth = int(run.parameters["maxDepth"]),
        min_samples_split = int(run.parameters["minSamplesSplit"]),
        criterion = 'entropy',
        random_state = 0
    )
    logging.info(f">> [Tabular Data Diagnostics] {classifier}")

    classifier.fit(xTrain, yTrain)

    logging.info(">> [Tabular Data Diagnostics] Starting with predictions...")

    yPred = classifier.predict(xTest)
    accuracy = accuracy_score(yTest, yPred)

    logging.info(f">> [Tabular Data Diagnostics] Model acc score: {accuracy_score(yTest, yPred)}")

    modelName = modelPath / "finalized_model.sav"
    validationDfPath = folder_manager.temp / "validation.csv"

    test = test.assign(Prediction = yPred)
    test.to_csv(validationDfPath, index = True)

    logging.info(f">> [Tabular Data Diagnostics] Creating validation.csv in artifacts...")
    run.createArtifact(validationDfPath, validationDfPath.name)

    with open(modelName, 'wb') as modelFile:
        pickle.dump(classifier, modelFile, -1)

    logging.info(f">> [Tabular Data Diagnostics] Saving model to Coretex...")
    saveModel(run, accuracy, xTrain.shape[1], labels)


if __name__ == "__main__":
    initializeJob(main)
