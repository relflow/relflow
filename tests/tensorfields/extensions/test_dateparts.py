from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.helpers import Jitter
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.tensorfields.base import TensorInput
from relflow.tensorfields.extensions.dateparts import DatePart, Embedder

ADDRESS = rf.Address("/", "created")


def build(jitter: Jitter | dict[str, object] | None = None) -> rf.Model:
    options = {} if jitter is None else {"jitter": jitter}
    return rf.Model.xs(created=rf.DateParts(dateparts=["day_of_week"], **options), d_model=8)


def test_dateparts_request_hydrates_jitter_from_a_mapping():
    model = build({"add": 0.2, "multiply": 0.1, "normalize": False})

    assert model.schema.requests[ADDRESS].jitter == Jitter(add=0.2, multiply=0.1, normalize=False)


@pytest.mark.parametrize("value", [None, 0.0, 0.2, 1, True])
def test_dateparts_rejects_non_jitter_configuration(value: object):
    with pytest.raises(pydantic.ValidationError):
        rf.DateParts(dateparts=["day_of_week"], jitter=value)


@pytest.mark.parametrize("normalize", [True, False])
def test_dateparts_jitters_its_only_continuous_boundary(
    monkeypatch: pytest.MonkeyPatch,
    normalize: bool,
):
    configured = Jitter(add=0.2, normalize=normalize)
    model = build(configured)
    embedder: Embedder = model.nodes[ADDRESS].embedder
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self == configured
        captured.append((inputs.clone(), mask.clone()))
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)
    inputs = TensorInput(
        state=torch.tensor([Tokens.valued, Tokens.null, Tokens.masked, Tokens.valued]),
        content=TensorDict(
            {
                DatePart.day_of_week: torch.tensor(
                    [
                        [0.5, -0.5],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [float("inf"), 1.0],
                    ]
                )
            },
            batch_size=[4],
        ),
        batch_size=[4],
    )

    embedder.train()
    embedder(inputs)
    embedder.eval()
    embedder(inputs)

    assert len(captured) == 1
    content, eligible = captured[0]
    assert content.shape == (4, 2)
    assert eligible.tolist() == [
        [True, True],
        [False, False],
        [False, False],
        [False, True],
    ]


def test_dateparts_jitters_unit_circle_inputs_without_mutating_targets(monkeypatch: pytest.MonkeyPatch):
    configured = Jitter(add=0.2)
    model = rf.Model.xs(
        created=rf.DateParts(
            dateparts=["day_of_week"],
            jitter=configured,
            mask=rf.Mask(query="supervise", reconstruct=True),
        ),
        d_model=8,
    )
    embedder: Embedder = model.nodes[ADDRESS].embedder
    field = model.encode(
        pa.table(
            {
                "created": [datetime(2026, 5, 28, 14, 30, 45), datetime(2026, 5, 29, 14, 30, 45)],
                "supervise": [False, True],
            }
        ),
        strata=Strata.train,
    )[ADDRESS]
    content = field.content[DatePart.day_of_week].clone()
    targets = field.targets[TensorKey.content][DatePart.day_of_week].clone()
    assert content[0].count_nonzero() > 0
    assert targets[1].count_nonzero() > 0
    captured: list[torch.Tensor] = []

    def shift(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        captured.append(inputs.clone())
        return inputs + mask.to(dtype=inputs.dtype).mul(0.25)

    monkeypatch.setattr(Jitter, "apply", shift)
    embedder.train()
    embedder.embed(field)

    assert len(captured) == 1
    assert torch.equal(captured[0], content.reshape(-1, 2))
    assert torch.equal(field.content[DatePart.day_of_week], content)
    assert torch.equal(field.targets[TensorKey.content][DatePart.day_of_week], targets)


def test_default_dateparts_jitter_does_not_advance_rng_during_training():
    model = build()
    embedder: Embedder = model.nodes[ADDRESS].embedder
    field = model.encode(
        pa.table({"created": [datetime(2026, 5, 28, 14, 30, 45)]}),
        strata=Strata.train,
    )[ADDRESS]
    before = torch.random.get_rng_state().clone()

    embedder.train()
    embedder.embed(field)

    assert torch.equal(torch.random.get_rng_state(), before)


def recovered(pairs: torch.Tensor, period: int) -> list[int]:
    """Read calendar values back from their (sin, cos) pairs."""
    angles = np.arctan2(pairs[..., 0].numpy(), pairs[..., 1].numpy()) % (2 * np.pi)
    return [int(round(value)) % period for value in angles.reshape(-1) / (2 * np.pi) * period]


@pytest.mark.parametrize("zone", ["America/New_York", "Asia/Tokyo"])
def test_dateparts_reads_zone_aware_timestamps_in_their_own_zone(zone: str):
    tz = ZoneInfo(zone)
    stamps = [datetime(2024, 3, 9, 18, 30, tzinfo=tz), datetime(2024, 3, 9, 23, 30, tzinfo=tz)]
    model = rf.Model.xs(
        d_model=8, when=rf.DateParts(dateparts=["day_of_week", "hour_of_day"]), label=rf.Boolean(mask=True)
    )
    aware = model.encode(pa.table({"when": pa.array(stamps, type=pa.timestamp("s", tz=zone))}), strata="predict")
    naive = model.encode(pa.table({"when": pa.array([s.replace(tzinfo=None) for s in stamps])}), strata="predict")
    content = aware[rf.Address("when")].content
    expected = naive[rf.Address("when")].content

    hour = [key for key in content.keys() if "hour" in str(key)][0]
    week = [key for key in content.keys() if "week" in str(key)][0]
    assert recovered(content[hour], 24) == [18, 23]
    assert recovered(content[week], 7) == [6, 6]  # Saturday in both cases, Sunday being 0
    torch.testing.assert_close(content[hour], expected[hour])
    torch.testing.assert_close(content[week], expected[week])
