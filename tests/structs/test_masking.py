import typing

import pydantic
import pytest

import relflow as rf
from relflow.structs.masking import SelectorConfig
from relflow.structs.tree import Address, scopes


def test_selectors_round_trip_through_mask_json():
    mask = rf.Mask(select=rf.Spans(rate=0.15), reconstruct=True)
    restored = rf.Mask.model_validate_json(mask.model_dump_json())
    assert restored == mask
    assert restored.select == rf.Spans(rate=0.15, length=5)
    payload = {"select": {"type": "forecast", "horizon": 24, "targets": ["ot"]}, "reconstruct": True}
    assert rf.Mask.model_validate(payload).select == rf.Forecast(horizon=24, targets=("ot",))


@pytest.mark.parametrize("options", [{"rate": 0.0}, {"rate": 1.0}, {"rate": 0.2, "length": 0}])
def test_spans_reject_out_of_range_options(options):
    with pytest.raises(pydantic.ValidationError):
        rf.Spans(**options)


@pytest.mark.parametrize("horizon", [0, -1, 2.5, True])
def test_forecast_requires_a_positive_integer_horizon(horizon):
    with pytest.raises(pydantic.ValidationError):
        rf.Forecast(horizon=horizon, targets=("ot",))


def test_forecast_requires_at_least_one_target():
    with pytest.raises(pydantic.ValidationError):
        rf.Forecast(horizon=2, targets=())
    with pytest.raises(pydantic.ValidationError):
        rf.Forecast(horizon=2)


def test_scope_table_covers_every_selector_setting():
    union = typing.get_args(typing.get_args(SelectorConfig)[0])
    assert set(scopes) == {type(None), *union}


def owner(mask: rf.Mask) -> rf.Branch:
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=3,
            direction=rf.Direction.causal,
            time_index="ts",
            time_step=1.0,
            mask=mask,
            ts=rf.Number,
            ot=rf.Number,
            temperature=rf.Number,
            load=rf.Number,
            tags=rf.Branch(length=2, name=rf.Category),
        ),
    )
    return schema.branches["/events"]


def test_forecast_hides_all_but_time_index_and_known_and_supervises_targets():
    mask = rf.Mask(select=rf.Forecast(horizon=1, targets=("ot",), known=("temperature",)), reconstruct=True)
    events = owner(mask)
    hidden = {
        name: mask.hides(owner=events, leaf=Address("events", name)) for name in ("ts", "ot", "temperature", "load")
    }
    assert hidden == {"ts": False, "ot": True, "temperature": False, "load": True}
    assert mask.hides(owner=events, leaf=Address("events", "tags", "name"))
    assert mask.supervises(owner=events, leaf=Address("events", "ot"))
    assert not mask.supervises(owner=events, leaf=Address("events", "load"))
    assert not mask.supervises(owner=events, leaf=Address("events", "tags", "name"))


@pytest.mark.parametrize(
    "mask", [rf.Mask(select=rf.Spans(rate=0.2), reconstruct=True), rf.Mask(rate=0.5, reconstruct=True)]
)
def test_spans_and_coordinate_masks_hide_and_supervise_every_descendant(mask):
    events = owner(mask)
    for leaf in (Address("events", "ts"), Address("events", "tags", "name")):
        assert mask.hides(owner=events, leaf=leaf)
        assert mask.supervises(owner=events, leaf=leaf)
