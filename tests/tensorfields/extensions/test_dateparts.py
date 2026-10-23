import math
from datetime import datetime

import numpy as np
import pyarrow as pa
import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.helpers import Jitter
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Prediction
from relflow.tensorfields.base import TensorInput
from relflow.tensorfields.extensions.dateparts import DatePart, Embedder

ADDRESS = rf.Address("/", "created")


def build(jitter: Jitter | dict[str, object] | None = None) -> rf.Model:
    options = {} if jitter is None else {"jitter": jitter}
    return rf.Model.xs(created=rf.DateParts(dateparts=["day_of_week"], **options), d_model=8)


def test_dateparts_request_hydrates_jitter_from_a_mapping():
    model = build({"add": 0.2, "multiply": 0.1, "normalize": False})

    assert model.schema.requests[ADDRESS].jitter == Jitter(add=0.2, multiply=0.1, normalize=False)


def test_dateparts_content_loss_and_angle_exclude_null_targets(monkeypatch: pytest.MonkeyPatch):
    model = rf.Model.xs(created=rf.DateParts(dateparts=["day_of_week"], mask=True), d_model=8)
    rows = pa.table({"created": pa.array([datetime(2024, 1, 1), None, datetime(2024, 1, 4)])})
    field = model.encode(rows, strata=Strata.train)[ADDRESS]
    target = field.targets[TensorKey.content]["day_of_week"].reshape(3, 2)
    # Predict the first target everywhere: exact for row 0, wrong for row 2, undefined for the null row.
    content = target[0].expand(3, 2).clone()
    state_logits = torch.full((3, len(Tokens)), -50.0)
    state_logits.scatter_(-1, field.targets[TensorKey.state].reshape(3, 1), 50.0)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.content: TensorDict({"day_of_week": content}, batch_size=[3])},
            batch_size=[3],
        ),
    )
    tracked = {}
    monkeypatch.setattr(model, "track", lambda names, value, **options: tracked.setdefault(names[2:], value))

    rf.TENSORFIELDS["dateparts"].loss(model, prediction, field, Strata.train)

    angle = 2 * math.pi * 3 / 7
    assert tracked[Metric.loss, TensorKey.content, "day_of_week"].item() == pytest.approx((1 - math.cos(angle)) / 2)
    assert tracked[Metric.mae, TensorKey.content, "day_of_week"].item() == pytest.approx(angle / 2)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_dateparts_evaluation_angles_count_valued_targets_across_unequal_batches(batch_size, tmp_path):
    model = rf.Model.xs(created=rf.DateParts(dateparts=["day_of_week"], mask=True), d_model=8)
    decoder = model.nodes[ADDRESS].decoder
    with torch.no_grad():
        # A constant readout on the unit circle: day 0 of the Sunday-based week.
        decoder.dateparts["day_of_week"].weight.zero_()
        decoder.dateparts["day_of_week"].bias.copy_(torch.tensor([0.0, 1.0]))
    rows = pa.table({"created": pa.array([datetime(2024, 1, 7), datetime(2024, 1, 10), None])})
    data = rf.ArrowDataModule(model, validate=rows)
    values = model.validate(
        data,
        batch_size=batch_size,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )[0]
    # Sunday is exact and Wednesday is three sevenths of a turn away; the null row adds nothing.
    assert values[".created/validate.mae.content.day_of_week"] == pytest.approx(2 * math.pi * 3 / 7 / 2, rel=1e-4)


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


@pytest.mark.parametrize(
    ("zone", "utc"),
    [
        ("America/New_York", ["2024-03-09T23:30", "2024-03-10T04:30"]),
        ("Asia/Tokyo", ["2024-03-09T09:30", "2024-03-09T14:30"]),
    ],
)
def test_dateparts_reads_zone_aware_timestamps_in_their_own_zone(zone: str, utc: list[str]):
    # The same two local wall-clock times, 18:30 and 23:30 on Saturday 2024-03-09, as instants in each zone.
    instants = pa.array(np.array(utc, dtype="datetime64[s]")).cast(pa.timestamp("s", tz=zone))
    local = pa.array(np.array(["2024-03-09T18:30", "2024-03-09T23:30"], dtype="datetime64[s]"))
    model = rf.Model.xs(
        d_model=8, when=rf.DateParts(dateparts=["day_of_week", "hour_of_day"]), label=rf.Boolean(mask=True)
    )
    aware = model.encode(pa.table({"when": instants}), strata="predict")
    naive = model.encode(pa.table({"when": local}), strata="predict")
    content = aware[rf.Address("when")].content
    expected = naive[rf.Address("when")].content

    hour = [key for key in content.keys() if "hour" in str(key)][0]
    week = [key for key in content.keys() if "week" in str(key)][0]
    assert recovered(content[hour], 24) == [18, 23]
    assert recovered(content[week], 7) == [6, 6]  # Saturday in both cases, Sunday being 0
    torch.testing.assert_close(content[hour], expected[hour])
    torch.testing.assert_close(content[week], expected[week])
