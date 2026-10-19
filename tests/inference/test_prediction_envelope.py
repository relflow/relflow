import pyarrow as pa
import pytest
import torch

import relflow as rf


def test_prediction_envelope_includes_target_state_and_normalized_embedding() -> None:
    model = rf.Model.xs(
        d_model=8,
        embed=True,
        amount=rf.Number,
        label=rf.Category(mask=True, p_unavailable=0.0),
    )
    model.encode(
        pa.table({"amount": [1.0, 2.0], "label": ["no", "yes"]}),
        strata="train",
    )

    output = model.predict(pa.table({"amount": [1.5]}))
    predictions = output["predictions"].combine_chunks()
    target = predictions.field("label")

    assert {field.name for field in target.type} == {"state", "content", "inferred"}
    assert {field.name for field in target.type.field("state").type} == {token.name for token in rf.Tokens}
    assert {field.name for field in target.type.field("content").type} == {"value", "probability", "topk"}
    assert target.field("inferred").to_pylist() == [True]
    representation = output["representations"][0].as_py()[0]
    assert representation["path"] == []
    assert representation["parents"] == []
    assert representation["output_index"] == 0
    embedding = representation["embedding"]
    assert torch.linalg.vector_norm(torch.tensor(embedding)).item() == pytest.approx(1.0)
    assert output.schema.metadata == {
        b"relflow.prediction_format": b"2",
        b"relflow.model_version": model.version.encode("utf-8"),
    }
    assert "inputs" not in output.column_names


def test_embedding_only_leaf_does_not_enable_a_decoder_or_predictions_column() -> None:
    model = rf.Model.xs(d_model=8, merchant=rf.Category(embed=True))
    model.encode(pa.table({"merchant": ["shop", "cafe"]}), strata="train")

    output = model.predict(pa.table({"merchant": ["shop"]}))

    assert model.schema.decodes == []
    assert output.column_names == ["representations"]
    representation = output["representations"][0].as_py()[0]
    assert representation["path"] == ["merchant"]
    assert representation["parents"] == []
    assert representation["output_index"] == 0
    assert len(representation["embedding"]) == model.schema.d_model
