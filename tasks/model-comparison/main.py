import logging

from coretex import currentTaskRun, Model


def main() -> None:
    taskRun = currentTaskRun()
    models: list[Model] = taskRun.parameters["models"]

    logging.info(">> [Model Comparison] Searching for most accurate model...")
    bestModel = max(models, key = lambda x: x.accuracy)
    logging.info(f">> [Model Comparison] Most accurate model found \"{bestModel.name}\"")

    logging.info(">> [Model Comparison] Submitting model as output.")
    taskRun.submitOutput("outputModel", bestModel)
    logging.info(">> [Model Comparison] Model submitted!")


if __name__ == "__main__":
    main()
