from coretex import currentTaskRun, Metric, MetricType
from ultralytics.models.yolo.detect.train import DetectionTrainer


def onTrainStart(trainer: DetectionTrainer) -> None:
    currentTaskRun().createMetrics([
        Metric.create("precision", "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("recall",    "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("mAP50",     "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("mAP50-95",  "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("box_loss",  "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("cls_loss",  "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("dfl_loss",  "epoch", MetricType.int, "value", MetricType.float),
        Metric.create("seg_loss",  "epoch", MetricType.int, "value", MetricType.float)
    ])


def onEpochEnd(trainer: DetectionTrainer) -> None:
    if not isinstance(trainer.metrics, dict):
        return

    metrics: dict[str, tuple[float, float]] = {}

    for key, value in trainer.metrics.items():
        name: str = key.split("/")[-1]
        if name.endswith("(B)"):
            name = name[:-3]

        if name.endswith("(M)"):
            name = name[:-3]

        metrics[name] = (trainer.epoch + 1, value)

    currentTaskRun().submitMetrics(metrics)
