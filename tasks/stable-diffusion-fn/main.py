import logging

from coretex import currentTaskRun, Model


def main():
    taskRun = currentTaskRun()

    logging.info(">> [StableDiffusion] Creating Coretex model")
    model = Model.createModel(f"{taskRun.id}-{taskRun.name}", taskRun.projectId, 1.0)

    logging.info(">> [StableDiffusion] Uploading files")
    model.upload("resources")

    taskRun.submitOutput("outputModel", model)


if __name__ == "__main__":
    main()
