from coretex import currentTaskRun, Model

def main() -> None:
    taskRun = currentTaskRun()

    model = Model.createModel(taskRun.generateEntityName(), taskRun.projectId, 1.0, {})
    model.upload("resources")

    taskRun.submitOutput("outputModel", model)


if __name__ == "__main__":
    main()
