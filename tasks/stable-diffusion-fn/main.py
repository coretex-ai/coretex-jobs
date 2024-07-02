import logging

from coretex import currentTaskRun, Model


def main() -> None:
    taskRun = currentTaskRun()

    logging.info(">> [StableDiffusion] Creating Coretex model")
    model = Model.createModel(taskRun.generateEntityName(), taskRun.id, 1.0, {})

    logging.info(">> [StableDiffusion] Uploading files")
    model.upload("resources")


if __name__ == "__main__":
    main()
