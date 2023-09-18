import inspect
import sys
import logging

from coretex import _task
from coretex.networking import RequestFailedError


if __name__ == "__main__":
    try:
        experimentId, callback = _task._processRemote(sys.argv)
    except:
        experimentId, callback = _task._processLocal(sys.argv)

    try:
        experiment = _task._prepareForExecution(experimentId)
        _task._current_experiment.setCurrentExperiment(experiment)

        callback.onStart()

        logging.getLogger("coretexpylib").info("Experiment execution started")
        exec(inspect.getsource(__import__("main")))

        callback.onSuccess()
    except RequestFailedError:
        callback.onNetworkConnectionLost()
    except KeyboardInterrupt:
        callback.onKeyboardInterrupt()
    except BaseException as ex:
        callback.onException(ex)

        # sys.exit is ok here, finally block is guaranteed to execute
        # due to how sys.exit is implemented (it internally raises SystemExit exception)
        sys.exit(1)
    finally:
        callback.onCleanUp()
