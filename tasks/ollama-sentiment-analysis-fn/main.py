from coretex import currentTaskRun, Model

def main():
    taskRun = currentTaskRun()

    model = Model.createModel(f"{taskRun.id}-{taskRun.name}", taskRun.projectId, 1.0)
    model.upload("resources")

    taskRun.submitOutput("outputModel", model)


if __name__ == "__main__":
    main()
