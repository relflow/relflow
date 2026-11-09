import polars as pl
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
    assert isinstance(output, pl.DataFrame)
    target = output["predictions"].struct.field("label")

    assert set(target.struct.fields) == {"state", "content", "inferred"}
    assert set(target.struct.field("state").struct.fields) == {token.name for token in rf.Tokens}
    assert set(target.struct.field("content").struct.fields) == {"value", "probability", "topk"}
    assert target.struct.field("inferred").to_list() == [True]
    representation = output["representations"].to_list()[0][0]
    assert representation["path"] == []
    assert representation["parents"] == []
    assert representation["output_index"] == 0
    embedding = representation["embedding"]
    assert torch.linalg.vector_norm(torch.tensor(embedding)).item() == pytest.approx(1.0)
    assert "inputs" not in output.columns
    assert output["representations"].explode().struct.field("embedding").dtype == pl.Array(pl.Float32, 8)


def test_embedding_only_leaf_emits_representations_without_predictions() -> None:
    model = rf.Model.xs(d_model=8, merchant=rf.Category(embed=True))
    model.encode(pa.table({"merchant": ["shop", "cafe"]}), strata="train")

    output = model.predict(pa.table({"merchant": ["shop"]}))

    assert model.schema.decodes == []
    assert output.columns == ["representations"]
    representation = output["representations"].to_list()[0][0]
    assert representation["path"] == ["merchant"]
    assert representation["parents"] == []
    assert representation["output_index"] == 0
    assert len(representation["embedding"]) == model.schema.d_model


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("kind", ["arrow", "polars"])
def test_model_write_returns_polars_with_the_same_prediction_contract(empty, kind) -> None:
    model = rf.Model.xs(d_model=8, embed=True, amount=rf.Number, label=rf.Boolean(mask=True)).eval()
    source = pa.table({"amount": [1.5], "id": [7]})
    if empty:
        source = source.slice(0, 0)
    if kind == "polars":
        source = pl.from_arrow(source)
    with torch.inference_mode():
        predictions = [] if empty else model(model.encode(source), strata="predict")
        result = model.write(predictions, source=source, retain=("id",))
    assert isinstance(result, pl.DataFrame)
    expected = model.predict(source, retain=("id",))
    assert result.schema == expected.schema
    assert result.equals(expected)


def test_direct_prediction_returns_the_final_postprocessor_frame() -> None:
    model = rf.Model.xs(d_model=8, amount=rf.Number, label=rf.Boolean(mask=True))
    outputs = []
    marker = object()

    @rf.postprocess
    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        result = pl.DataFrame({"marker": pl.Series([marker] * frame.height, dtype=pl.Object)})
        outputs.append(result)
        return result

    result = model.predict(pa.table({"amount": [1.5, 2.5]}), batch_size=1, postprocess=finish)
    assert result is outputs[0]
    assert result["marker"].to_list() == [marker, marker]
