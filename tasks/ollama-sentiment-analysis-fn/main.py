from coretex import currentTaskRun, Model

def main():
    taskRun = currentTaskRun()

    model = Model.createModel(taskRun.generateEntityName(), taskRun.id, 1.0, {})
    model.upload("resources")


if __name__ == "__main__":
    main()
