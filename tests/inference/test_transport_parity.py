from __future__ import annotations

import asyncio
from types import SimpleNamespace

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import relflow as rf
from relflow.inference.deployment import ARROW_MEDIA_TYPE, JSON_MEDIA_TYPE


@pytest.mark.parametrize("retain", [(), "*"])
def test_direct_parquet_arrow_and_json_preserve_the_same_selected_columns(tmp_path, retain):
    model = rf.Model.xs(
        d_model=8,
        embed=True,
        amount=rf.Number,
        disputed=rf.Boolean(mask=True),
    )
    source = pa.table({"amount": [1.0, 2.0]})
    direct = model.predict(source, retain=retain, batch_size=2)
    writer = rf.Writer(tmp_path)
    writer.write_on_batch_end(
        SimpleNamespace(global_rank=0),
        SimpleNamespace(),
        direct,
        None,
        None,
        0,
        0,
    )
    writer.close()
    persisted = pq.read_table(tmp_path / "rank-0.parquet")
    assert persisted.equals(direct)
    assert persisted.schema.equals(direct.schema, check_metadata=True)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, source.schema) as stream:
        stream.write_table(source)
    arrow_body = sink.getvalue().to_pybytes()
    app = rf.Deployment(model=model, accelerator="cpu", max_batch_size=2, retain=retain).app()
    endpoint = next(route.endpoint for route in app.routes if route.path == "/predict")

    class Request:
        def __init__(self, media_type, body):
            self.headers = {"content-type": media_type}
            self.content = body

        async def body(self):
            return self.content

    async def run():
        async with app.router.lifespan_context(app):
            arrow = await endpoint(Request(ARROW_MEDIA_TYPE, arrow_body))
            json = await endpoint(Request(JSON_MEDIA_TYPE, orjson.dumps(source.to_pylist())))
            rows = source.to_pylist()
            partial = await endpoint(Request(JSON_MEDIA_TYPE, orjson.dumps([rows[0], "invalid", rows[1]])))
            return arrow, json, partial

    arrow_response, json_response, partial_response = asyncio.run(run())
    served = pa.ipc.open_stream(pa.BufferReader(arrow_response.body)).read_all()
    assert served.equals(direct)
    assert served.schema.equals(direct.schema, check_metadata=True)
    assert orjson.loads(json_response.body) == direct.to_pylist()
    assert "model" not in served.column_names
    assert ("inputs" in served.column_names) == (retain == "*")
    partial = orjson.loads(partial_response.body)
    assert len(partial) == 3
    assert set(partial[0]) == set(direct.to_pylist()[0])
    assert partial[1] == {
        "predictions": None,
        "error": {"status_code": 422, "message": "each request must be a JSON object, got str"},
    }
    assert set(partial[2]) == set(direct.to_pylist()[1])
    for response in (arrow_response, json_response, partial_response):
        assert response.headers["x-relflow-model-version"] == model.version
