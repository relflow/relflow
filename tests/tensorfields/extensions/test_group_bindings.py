"""Tests for Group runtime state accessors."""

from __future__ import annotations

import pytest
import torch

import relflow as rf
from relflow.tensorfields.extensions.group import Embedder
from tests.arrow import table

ADDRESS = rf.Address("/merchant_id")
N_GROUPS = 3


def build() -> rf.Model:
    torch.manual_seed(0)
    return rf.Model.xs(
        d_model=8,
        batch_size=2,
        attention=None,
        merchant_id=rf.Group(
            n_groups=3,
            mask=rf.Mask(rate=0.1, reconstruct=True),
            p_unavailable=0.0,
        ),
    )


def embedder(model: rf.Model) -> Embedder:
    embedder = model.nodes[ADDRESS].embedder
    assert isinstance(embedder, Embedder)
    return embedder


def learn(model: rf.Model, *tokens: str) -> None:
    model.encode(
        table([{"merchant_id": token} for token in tokens]),
        strata=rf.Strata.train,
    )


def test_group_vocabulary_returns_immutable_model_snapshot() -> None:
    model = build()
    learn(model, "ALPHA", "BETA")

    snapshot = rf.Group.vocabulary(model, ADDRESS)

    assert snapshot == ("ALPHA", "BETA")
    assert isinstance(snapshot, tuple)

    learn(model, "GAMMA")

    assert snapshot == ("ALPHA", "BETA")
    assert rf.Group.vocabulary(model, "/merchant_id") == (
        "ALPHA",
        "BETA",
        "GAMMA",
    )


def test_group_vocabulary_reads_latest_encoding_context_snapshot() -> None:
    model = build()
    encoding_context = model.interprocess_encoding_context

    learn(model, "ALPHA", "BETA")

    assert rf.Group.vocabulary(encoding_context, ADDRESS) == ("ALPHA", "BETA")


def test_group_vocabulary_reads_same_length_context_replacement() -> None:
    model = build()
    learn(model, "ALPHA")
    encoding_context = model.interprocess_encoding_context

    embedder(model).vocab.load_snapshot(["BETA"])

    assert rf.Group.vocabulary(encoding_context, ADDRESS) == ("BETA",)


def test_group_assignments_return_immutable_snapshots_without_sentinel() -> None:
    model = build()
    rf.Group.assign(model, ADDRESS, "ALPHA", 2)
    rf.Group.assign(model, ADDRESS, "BETA", 0)

    snapshot = rf.Group.assignments(model, ADDRESS)

    assert tuple(snapshot) == ("ALPHA", "BETA")
    assert snapshot["ALPHA"]["group"] == 2
    assert snapshot["BETA"]["group"] == 0
    assert len(snapshot["ALPHA"]["probabilities"]) == N_GROUPS
    assert sum(snapshot["ALPHA"]["probabilities"]) == pytest.approx(1.0)
    expected = torch.tensor([0.0, 0.0, 1.0]).softmax(dim=-1).tolist()
    assert snapshot["ALPHA"]["probabilities"] == pytest.approx(expected)

    rf.Group.assign(model, ADDRESS, "ALPHA", 0)

    assert snapshot["ALPHA"]["group"] == 2
    assert rf.Group.assignments(model, ADDRESS)["ALPHA"]["group"] == 0


def test_group_assignments_empty_vocabulary_is_empty() -> None:
    assert rf.Group.assignments(build(), ADDRESS) == {}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_group_assignments_use_the_model_device(device: str) -> None:
    model = build().to(device)

    index = rf.Group.assign(model, ADDRESS, "ALPHA", 2)

    assert embedder(model).embeddings["group"].weight[index].device.type == device
    assert rf.Group.assignments(model, ADDRESS)["ALPHA"]["group"] == 2


def test_group_assignments_preserve_source_precision_near_tied_logits() -> None:
    model = build().double()
    learn(model, "ALPHA")
    weight = embedder(model).embeddings["group"].weight
    with torch.no_grad():
        weight[0].copy_(torch.tensor([1.0, 1.0 + 1e-8, 0.0], dtype=torch.float64))

    snapshot = rf.Group.assignments(model, ADDRESS)["ALPHA"]

    assert snapshot["group"] == 1
    expected = weight[0].detach().softmax(dim=-1).tolist()
    assert snapshot["probabilities"] == pytest.approx(expected, abs=1e-12)


def test_group_decoder_does_not_condition_identity_on_row_siblings() -> None:
    model = rf.Model.xs(
        d_model=8,
        merchant_id=rf.Group(
            n_groups=N_GROUPS,
            mask=rf.Mask(rate=0.1, reconstruct=True),
        ),
        amount=rf.Number,
    )

    decoder = model.nodes[ADDRESS].decoder

    assert decoder.context_addresses == ()
    assert decoder.context_projection is None


@pytest.mark.parametrize(
    "binding",
    [rf.Group.vocabulary, rf.Group.assignments],
    ids=["vocabulary", "assignments"],
)
def test_group_model_bindings_reject_missing_address(binding) -> None:
    with pytest.raises(KeyError, match="missing"):
        binding(build(), "/missing")


@pytest.mark.parametrize(
    "binding",
    [rf.Group.vocabulary, rf.Group.assignments],
    ids=["vocabulary", "assignments"],
)
def test_group_model_bindings_reject_non_group_field(binding) -> None:
    model = rf.Model.xs(
        d_model=8,
        amount=rf.Number,
    )

    with pytest.raises(TypeError, match="not a Group field"):
        binding(model, "/amount")


def test_group_vocabulary_validates_context_resource_and_source() -> None:
    with pytest.raises(TypeError, match="VocabularyState"):
        rf.Group.vocabulary({ADDRESS: object()}, ADDRESS)

    with pytest.raises(TypeError, match="Model or InterprocessEncodingContext"):
        rf.Group.vocabulary(object(), ADDRESS)


def test_group_model_only_bindings_validate_source() -> None:
    with pytest.raises(TypeError, match="must be a Model"):
        rf.Group.assignments(object(), ADDRESS)
