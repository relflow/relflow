from types import SimpleNamespace

import pytest
import torch

from relflow.logging.throughput import ThroughputLogger
from relflow.structs.enums import Metric, Strata


@pytest.mark.parametrize(
    ("strata", "stage"),
    [(Strata.train, "train"), (Strata.validate, "validation"), (Strata.test, "test")],
)
def test_throughput_logs_each_batch_without_epoch_aggregation(monkeypatch, strata, stage):
    tracked = []
    clock = iter([10.0, 12.0, 20.0, 24.0])
    monkeypatch.setattr("relflow.logging.throughput.perf_counter", lambda: next(clock))

    def track(names, value, **kwargs):
        tracked.append((names, value.item(), kwargs))

    callback = ThroughputLogger()
    trainer = SimpleNamespace(global_step=0)
    module = SimpleNamespace(batch_size=10, device=torch.device("cpu"), track=track)
    for batch_idx in range(2):
        getattr(callback, f"on_{stage}_batch_start")(trainer, module, None, batch_idx)
        getattr(callback, f"on_{stage}_batch_end")(trainer, module, None, None, batch_idx)
    getattr(callback, f"on_{stage}_epoch_end")(trainer, module)

    assert tracked == [
        ((Metric.throughput, strata), 5.0, {"on_step": True, "on_epoch": False}),
        ((Metric.throughput, strata), 2.5, {"on_step": True, "on_epoch": False}),
    ]
    assert callback.throughput[strata] == 2.5


@pytest.mark.parametrize("is_global_zero", [True, False])
def test_prediction_throughput_logs_each_batch_directly_on_rank_zero(monkeypatch, is_global_zero):
    clock = iter([10.0, 12.0, 20.0, 24.0])
    monkeypatch.setattr("relflow.logging.throughput.perf_counter", lambda: next(clock))

    class Logger:
        def __init__(self):
            self.records = []

        def log_metrics(self, metrics, step=None):
            self.records.append((metrics, step))

    def track(*args, **kwargs):
        raise AssertionError("prediction hooks must not call Model.track")

    callback = ThroughputLogger()
    loggers = [Logger(), Logger()]
    trainer = SimpleNamespace(loggers=loggers, is_global_zero=is_global_zero)
    module = SimpleNamespace(batch_size=10, track=track)
    for batch_idx in range(2):
        callback.on_predict_batch_start(trainer, module, None, batch_idx)
        callback.on_predict_batch_end(trainer, module, None, None, batch_idx)
    callback.on_predict_epoch_end(trainer, module)

    assert callback.throughput[Strata.predict] == 2.5
    expected = [({"throughput/predict": 5.0}, 0), ({"throughput/predict": 2.5}, 1)] if is_global_zero else []
    assert all(logger.records == expected for logger in loggers)
