import logging
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from coretex import CustomDataset, Experiment, Model, folder_manager, currentExperiment

from src.dataset import extractTestTrainData, loadDataset


def saveModel(experiment: Experiment[CustomDataset], accuracy: float, trainColumnCount: int, labels: list[str]):
    model = Model.createModel(experiment.name, experiment.id, accuracy, {})
    modelPath = folder_manager.temp / "model"

    model.saveModelDescriptor(modelPath, {
        "project_task": experiment.spaceTask,
        "labels": labels,
        "modelName": model.name,
        "description": experiment.description,

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


def main() -> None:
    experiment: Experiment[CustomDataset] = currentExperiment()

    modelPath = folder_manager.createTempFolder("model")

    train, test, labels = loadDataset(
        experiment.dataset,
        experiment.parameters["validationSplit"],
        experiment.parameters["labelColumn"],
        experiment.parameters["excludeColumns"]
    )

    xTrain, xTest, yTrain, yTest = extractTestTrainData(
        train,
        test,
        experiment.parameters["labelColumn"]
    )

    logging.info(">> [Tabular Data Diagnostics] Starting with training the model...")

    classifier = RandomForestClassifier(
        n_estimators = int(experiment.parameters["nEstimators"]),
        max_depth = int(experiment.parameters["maxDepth"]),
        min_samples_split = int(experiment.parameters["minSamplesSplit"]),
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
    experiment.createArtifact(validationDfPath, validationDfPath.name)

    with open(modelName, 'wb') as modelFile:
        pickle.dump(classifier, modelFile, -1)

    logging.info(f">> [Tabular Data Diagnostics] Saving model to Coretex...")
    saveModel(experiment, accuracy, xTrain.shape[1], labels)


if __name__ == "__main__":
    main()
