from coretex import currentTaskRun, Metric, MetricType
from ultralytics.models.yolo.detect.train import DetectionTrainer


def onTrainStart(trainer: DetectionTrainer) -> None:
    currentTaskRun().createMetrics([
        Metric.create("precision", "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs], [0, 1]),
        Metric.create("recall",    "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs], [0, 1]),
        Metric.create("mAP50",     "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs], [0, 1]),
        Metric.create("mAP50-95",  "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs], [0, 1]),
        Metric.create("box_om",    "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs]),
        Metric.create("cls_om",    "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs]),
        Metric.create("dfl_om",    "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs]),
        Metric.create("box_oo",    "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs]),
        Metric.create("cls_oo",    "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs]),
        Metric.create("dfl_oo",    "epoch", MetricType.int, "value", MetricType.float, [1, trainer.epochs])
    ])


def onEpochEnd(trainer: DetectionTrainer) -> None:
    if not isinstance(trainer.metrics, dict):
        return

    metrics: dict[str, tuple[float, float]] = {}

    for key, value in trainer.metrics.items():
        name: str = key.split("/")[-1]
        if name.endswith("(B)"):
            name = name[:-3]

        metrics[name] = (trainer.epoch + 1, value)

    currentTaskRun().submitMetrics(metrics)
