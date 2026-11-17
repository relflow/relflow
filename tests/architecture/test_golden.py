import json
import platform
from pathlib import Path

import polars as pl
import polars.testing
import pyarrow as pa
import pytest
import torch
from lightning.pytorch import seed_everything

import relflow as rf

FIXTURES = Path(__file__).parent / "golden"


def line(sku: str | None, quantity: float | None) -> dict:
    return {"sku": sku, "quantity": quantity}


def batches() -> dict[str, pa.Table]:
    mixed = [
        {
            "amount": 12.5,
            "merchant": "grocer",
            "label": "ok",
            "events": [
                {"kind": "buy", "value": 1.0, "lines": []},
                {
                    "kind": "buy",
                    "value": 2.0,
                    "lines": [line("a", 1.0), line("b", 2.0), line("c", 3.0), line("a", 4.0)],
                },
                {"kind": "sell", "value": 3.0, "lines": [line("a", 1.0)]},
                {"kind": "buy", "value": 4.0, "lines": [line("b", 2.0), None, line("c", 5.0)]},
                {"kind": "sell", "value": 5.0, "lines": [line("c", 1.0), line("a", 2.0)]},
            ],
            "history": [{"price": 1.5}, {"price": 2.5}, {"price": 3.5}],
            "notes": [{"score": 0.5}, {"score": 1.5}],
        },
        {"amount": -3.0, "merchant": "cafe", "label": "bad", "events": [], "history": None, "notes": [{"score": None}]},
        {"amount": 7.25, "merchant": None, "label": "ok", "events": None, "history": [], "notes": [{"score": 2.0}]},
    ]
    full = [
        {
            "amount": 1.0 + index,
            "merchant": "grocer" if index else "cafe",
            "label": "ok" if index else "bad",
            "events": [
                {"kind": "buy", "value": 1.0 + step, "lines": [line("a", 1.0), line("b", 2.0), line("c", 3.0)]}
                for step in range(4)
            ],
            "history": [{"price": 1.0 + index}, {"price": 2.0 + index}],
            "notes": [{"score": 0.25 * index}, {"score": 0.75}],
        }
        for index in range(2)
    ]
    return {"mixed": pa.Table.from_pylist(mixed), "full": pa.Table.from_pylist(full)}


def platform_key() -> dict[str, str]:
    return {"torch": torch.__version__, "system": platform.system(), "machine": platform.machine()}


def assert_frames(actual: pl.DataFrame, expected: pl.DataFrame, *, exact: bool) -> None:
    polars.testing.assert_frame_equal(actual, expected, check_exact=exact, rel_tol=1e-5, abs_tol=1e-6)


def recorded() -> dict:
    return torch.load(FIXTURES / "model.pt", weights_only=False, map_location="cpu")


def exact() -> bool:
    return json.loads((FIXTURES / "meta.json").read_text()) == platform_key()


def test_golden_yaml_builds_the_recorded_schema():
    model = rf.Model.from_yaml(FIXTURES / "model.yaml")
    assert model.schema.model_dump(mode="json") == recorded()["schema"]


def test_golden_yaml_initializes_the_recorded_parameters():
    seed_everything(0, workers=True)
    state = rf.Model.from_yaml(FIXTURES / "model.yaml").state_dict()
    initial = torch.load(FIXTURES / "initial.pt", weights_only=True)
    tolerance = {"rtol": 0, "atol": 0} if exact() else {"rtol": 1e-5, "atol": 1e-6}
    assert list(state) == list(initial)
    for key, value in initial.items():
        torch.testing.assert_close(state[key], value, **tolerance)


def test_golden_checkpoint_loads_the_recorded_state():
    saved = recorded()["state_dict"]
    state = rf.Model.load(FIXTURES / "model.pt").state_dict()
    assert list(state) == list(saved)
    for key, value in saved.items():
        match value:
            case torch.Tensor():
                torch.testing.assert_close(state[key], value, rtol=0, atol=0)
            case _:
                assert state[key] == value


@pytest.mark.parametrize("name", ["mixed", "full"])
def test_golden_checkpoint_predicts_the_recorded_outputs(name):
    expected = pl.read_ipc(FIXTURES / f"{name}.arrow")
    actual = rf.Model.load(FIXTURES / "model.pt").predict(batches()[name])
    assert_frames(actual, expected, exact=exact())
