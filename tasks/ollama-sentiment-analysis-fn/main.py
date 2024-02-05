from coretex import currentTaskRun, Model

def main():
    taskRun = currentTaskRun()

    model = Model.createModel(taskRun.name, taskRun.id, 2.0, {})
    model.upload("resources")


if __name__ == "__main__":
    main()
