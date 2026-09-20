"""Structural controls for the isolated Category identity-channel experiment."""

import pyarrow as pa
import pytest
import torch
from experiments.category_hash import ARMS, Embedder, variant

import relflow as rf
from relflow.structs.enums import Component
from relflow.tensorfields.extensions.category import category
from relflow.tensorfields.extensions.hashable import hashable


def paired():
    return rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        left=rf.Category(p_unavailable=0.0, query="value"),
        right=rf.Category(p_unavailable=0.0, query="value"),
        fingerprint=rf.Hash(n_hashes=4, query="value"),
    )


@pytest.mark.parametrize("values", [["A", "B", "A"], [11, 12, 11], [b"A", b"B", b"A"]])
def test_fingerprints_match_native_hash_across_fields_and_stages(values):
    with variant(ARMS["sum-4"]):
        model = paired()
        rows = pa.table({"value": values})
        for stage in ("train", "validate", "test", "predict"):
            batch = model.encode(rows, strata=stage)
            left = batch["record/left"].hash
            assert torch.equal(left, batch["record/right"].hash)
            assert torch.equal(left, batch["record/fingerprint"].content)
            assert torch.equal(left[0], left[2])
            assert not torch.equal(left[0], left[1])
        first = model.encode(rows, strata="train")["record/left"].hash
        second = model.encode(rows, strata="train")["record/left"].hash
        assert not torch.equal(first, second)
        for stage in ("validate", "test", "predict"):
            first = model.encode(rows, strata=stage)["record/left"].hash
            second = model.encode(rows, strata=stage)["record/left"].hash
            assert torch.equal(first, second)


def test_dictionary_encoded_atoms_have_the_same_fingerprint():
    with variant(ARMS["sum-4"]):
        model = paired()
        values = pa.array(["A", "B", "A"])
        plain = model.encode(pa.table({"value": values}), strata="predict")
        encoded = model.encode(pa.table({"value": values.dictionary_encode()}), strata="predict")
        assert torch.equal(plain["record/left"].hash, encoded["record/left"].hash)


@pytest.mark.parametrize("arm", ["sum-1", "sum-4", "project-4"])
def test_hash_exists_before_admission_and_admission_preserves_embeddings(arm):
    with variant(ARMS[arm]):
        model = rf.Model(d_model=16, n_layers=1, n_heads=4, code=rf.Category(p_unavailable=0.0))
        embedder = model.nodes["record/code"].embedder
        assert isinstance(embedder, Embedder)
        old = model.encode(pa.table({"code": ["old"]}), strata="train")["record/code"]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        embedder.embed(old).payload.sum().backward()
        optimizer.step()
        assert embedder.embeddings["content"].weight[0].abs().sum() > 0

        rows = pa.table({"code": ["new"]})
        unknown = model.encode(rows, strata="validate")["record/code"]
        assert unknown.content.eq(8).all()
        assert unknown.hash.ne(0).any()
        before = embedder.embed(unknown).payload.detach()
        model.encode(rows, strata="train")
        known = model.encode(rows, strata="validate")["record/code"]
        assert known.content.eq(1).all()
        assert torch.equal(unknown.hash, known.hash)
        torch.testing.assert_close(before, embedder.embed(known).payload, rtol=0, atol=0)


def test_unavailable_augmentation_keeps_hash_and_pristine_category_target():
    with variant(ARMS["sum-4"]):
        model = rf.Model(
            d_model=16,
            n_layers=1,
            n_heads=4,
            code=rf.Category(p_unavailable=1.0, mask=rf.Mask(query="hide", reconstruct=True)),
            reference=rf.Hash(n_hashes=4, query="code"),
        )
        batch = model.encode(pa.table({"code": ["A", "B"], "hide": [False, False]}), strata="train")
        field = batch["record/code"]
        assert field.content.eq(8).all()
        assert field.targets["content"].lt(8).all()
        assert torch.equal(field.hash, batch["record/reference"].content)


@pytest.mark.parametrize("arm", ["sum-1", "sum-4", "project-4"])
def test_masking_removes_fingerprint_and_hidden_label_poison_cannot_change_predictions(arm):
    with variant(ARMS[arm]):
        model = rf.Model(
            d_model=16,
            n_layers=1,
            n_heads=4,
            code=rf.Category(p_unavailable=0.0, mask=rf.Mask(query="hide", reconstruct=True)),
        )
        rows = pa.table({"code": ["A", "B"], "hide": [True, True]})
        field = model.encode(rows, strata="train")["record/code"]
        assert field.state.eq(rf.Tokens.masked).all()
        assert field.hash.eq(0).all()
        assert field.take(None).content["hash"].eq(0).all()
        before = model.predict(rows).to_pylist()
        poisoned = pa.table({"code": ["poisoned", "other"], "hide": [True, True]})
        assert model.predict(poisoned).to_pylist() == before


def test_null_padding_and_compact_row_selection_preserve_alignment():
    with variant(ARMS["sum-4"]):
        model = rf.Model(
            d_model=16,
            n_layers=1,
            n_heads=4,
            items=rf.Branch(length=3, code=rf.Category(p_unavailable=0.0)),
        )
        rows = pa.Table.from_pylist([{"items": [{"code": "A"}, {"code": None}]}, {"items": [{"code": "B"}]}])
        field = model.encode(rows, strata="predict")["record/items/code"]
        assert field.hash.masked_select(~field.state.eq(rf.Tokens.valued).unsqueeze(-1)).eq(0).all()
        indices = torch.tensor([3, 0, 1], dtype=torch.int64)
        selected = field.take(indices)
        assert torch.equal(selected.state, field.state.reshape(-1)[indices])
        assert torch.equal(selected.content["hash"], field.hash.reshape(-1, 4)[indices])
        selected.content["hash"].zero_()
        assert field.hash.ne(0).any()


def test_baseline_preserves_parameters_targets_and_predictions_exactly():
    def build():
        return rf.Model(
            d_model=16,
            n_layers=1,
            n_heads=4,
            code=rf.Category(p_unavailable=0.0),
            label=rf.Category(mask=True),
        )

    torch.manual_seed(119)
    ordinary = build()
    rows = pa.table({"code": ["A", "B"], "label": ["yes", "no"]})
    ordinary.encode(rows, strata="train")
    expected = ordinary.predict(rows).to_pylist()
    with variant(ARMS["baseline"]):
        torch.manual_seed(119)
        instrumented = build()
        instrumented.encode(rows, strata="train")
        for left, right in zip(ordinary.parameters(), instrumented.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert instrumented.predict(rows).to_pylist() == expected


@pytest.mark.parametrize("arm", ["sum-4", "project-4"])
def test_checkpoint_roundtrip_inside_the_same_experiment_context(tmp_path, arm):
    with variant(ARMS[arm]):
        model = rf.Model(
            d_model=16,
            n_layers=1,
            n_heads=4,
            code=rf.Category(p_unavailable=0.0),
            label=rf.Category(mask=True),
        )
        rows = pa.table({"code": ["A", "B"], "label": ["yes", "no"]})
        model.encode(rows, strata="train")
        expected = model.predict(rows).to_pylist()
        path = model.save(tmp_path / "hybrid.pt")
        loaded = rf.Model.load(path)
        assert loaded.predict(rows).to_pylist() == expected


@pytest.mark.parametrize("value", [True, 1.5])
def test_experiment_rejects_unsupported_hash_atoms_explicitly(value):
    with variant(ARMS["sum-4"]):
        model = rf.Model(d_model=16, n_layers=1, n_heads=4, code=rf.Category())
        with pytest.raises(TypeError, match="integer/string/binary"):
            model.encode(pa.table({"code": [value]}), strata="predict")


def test_registry_restoration_and_unchanged_category_objective():
    before = dict(category.components)
    native = dict(hashable.components)
    with pytest.raises(RuntimeError, match="sentinel"):
        with variant(ARMS["sum-4"]):
            for key in (Component.Decoder, Component.loss, Component.observe, Component.learn, Component.write):
                assert category.components[key] is before[key]
            raise RuntimeError("sentinel")
    assert category.components == before
    assert hashable.components == native
