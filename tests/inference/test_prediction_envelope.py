import pyarrow as pa
import pytest
import torch

import relflow as rf


def test_prediction_envelope_includes_target_state_and_normalized_embedding() -> None:
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=1,
        embed=True,
        amount=rf.Number,
        label=rf.Category(mask=True, size=2, p_unavailable=0.0),
    )
    model.encode(
        pa.table({"amount": [1.0, 2.0], "label": ["no", "yes"]}),
        strata="train",
    )

    output = model.predict(pa.table({"amount": [1.5]}))
    predictions = output["predictions"].combine_chunks()
    target = predictions.field("record/label")
    root = predictions.field("record")

    assert {field.name for field in target.type} == {"state", "content", "inferred"}
    assert {field.name for field in target.type.field("state").type} == {token.name for token in rf.Tokens}
    assert {field.name for field in target.type.field("content").type} == {"value", "probability", "topk"}
    assert target.field("inferred").to_pylist() == [True]
    embedding = root.field("embedding")[0].as_py()
    assert torch.linalg.vector_norm(torch.tensor(embedding)).item() == pytest.approx(1.0)
