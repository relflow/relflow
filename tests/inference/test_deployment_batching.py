from __future__ import annotations

import asyncio
from types import SimpleNamespace

import orjson
import polars as pl
import pyarrow as pa
import pydantic
import pytest
import torch

import relflow as rf
import relflow.inference.deployment as deployment_module
from relflow import Model, Number, where
from relflow.data.processors import apply
from relflow.inference.deployment import Deployment, ErrorItem


class PredictModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, source, *, preprocess, postprocess, retain, batch_size=None):
        self.calls.append(
            {
                "source": source,
                "preprocess": preprocess,
                "postprocess": postprocess,
                "retain": retain,
                "batch_size": batch_size,
            }
        )
        ids = source["id"].combine_chunks()
        inputs = pa.StructArray.from_arrays([ids], names=["id"])
        labels = pa.array([str(value) for value in ids.to_pylist()], type=pa.large_string())
        predictions = pa.StructArray.from_arrays([labels], names=["label"])
        result = pa.table({"predictions": predictions})
        if retain == "*" or "id" in retain:
            result = result.append_column("inputs", inputs)
        return apply(result, postprocess)


def runtime(model=None, **kwargs) -> deployment_module.FastAPIRuntime:
    server = deployment_module.FastAPIRuntime(
        checkpoint="unused",
        accelerator=deployment_module.Accelerator.cpu,
        **kwargs,
    )
    server.model = PredictModel() if model is None else model
    server.device = torch.device("cpu")
    return server


def test_fastapi_batcher_splits_ready_requests_by_max_batch_size():
    class Runtime:
        def __init__(self):
            self.started = False
            self.batches = []

        def setup(self):
            self.started = True

        def predict_payloads(self, payloads):
            self.batches.append([payload["id"] for payload in payloads])
            return [{"id": payload["id"]} for payload in payloads]

    async def run():
        server = Runtime()
        batcher = deployment_module.FastAPIBatcher(runtime=server, max_batch_size=2, batch_timeout=0.0)
        await batcher.start()
        try:
            responses = await batcher.submit_many([{"id": 1}, {"id": 2}, {"id": 3}])
        finally:
            await batcher.stop()
        return server, responses

    server, responses = asyncio.run(run())

    assert server.started is True
    assert server.batches == [[1, 2], [3]]
    assert responses == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_batcher_runs_processors_once_for_one_collated_arrow_batch():
    preprocessed = []
    postprocessed = []

    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        preprocessed.append(frame)
        return frame

    @rf.postprocess
    def compact(frame: pl.DataFrame) -> pl.DataFrame:
        postprocessed.append("compact")
        return frame.select(value=pl.col("inputs").struct.field("value"))

    @rf.postprocess
    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        postprocessed.append("finish")
        return frame.select("value")

    async def run():
        model = Model(
            value=Number,
            d_model=8,
            n_layers=1,
            n_heads=2,
            embed=True,
        )
        server = deployment_module.FastAPIRuntime(
            checkpoint=model,
            accelerator=deployment_module.Accelerator.cpu,
            preprocessor=prepare,
            postprocessor=[compact, finish],
            retain=("value",),
        )
        batcher = deployment_module.FastAPIBatcher(
            runtime=server,
            max_batch_size=3,
            batch_timeout=0.05,
        )
        await batcher.start()
        try:
            return await asyncio.gather(
                batcher.submit({"value": 3.0}),
                batcher.submit({"value": 1.0}),
                batcher.submit({"value": 2.0}),
            )
        finally:
            await batcher.stop()

    responses = asyncio.run(run())

    assert len(preprocessed) == 1
    assert postprocessed == ["compact", "finish"]
    assert isinstance(preprocessed[0], pl.DataFrame)
    assert preprocessed[0]["value"].to_list() == [3.0, 1.0, 2.0]
    assert responses == [{"value": 3.0}, {"value": 1.0}, {"value": 2.0}]


def test_runtime_converts_valid_requests_to_one_arrow_prediction_call():
    model = PredictModel()

    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    server = runtime(model=model, preprocessor=prepare, retain=("id",))
    outputs = server.predict_payloads([{"id": 3}, {"id": 1}])

    assert len(model.calls) == 1
    call = model.calls[0]
    assert isinstance(call["source"], pa.Table)
    assert call["source"].to_pylist() == [{"id": 3}, {"id": 1}]
    assert call["preprocess"] == (prepare,)
    assert isinstance(call["postprocess"], tuple)
    assert call["postprocess"] == ()
    assert call["retain"] == ("id",)
    assert outputs == [
        {"predictions": {"label": "3"}, "inputs": {"id": 3}},
        {"predictions": {"label": "1"}, "inputs": {"id": 1}},
    ]


def test_runtime_predicts_an_arrow_table_without_python_rows():
    model = PredictModel()
    server = runtime(model=model)

    output = server.predict_table(pa.table({"id": [3, 1]}), batch_size=2)

    assert output.to_pylist() == [
        {"predictions": {"label": "3"}},
        {"predictions": {"label": "1"}},
    ]
    assert len(model.calls) == 1
    assert model.calls[0]["batch_size"] == 2


def test_runtime_preserves_embedding_only_outputs_and_invalid_request_slots():
    vector = pa.list_(pa.field("element", pa.float32(), nullable=False), 4)
    representation = pa.struct(
        [
            ("path", pa.list_(pa.field("element", pa.string()))),
            ("parents", pa.list_(pa.field("element", pa.int32()))),
            ("output_index", pa.int32()),
            pa.field("embedding", vector, nullable=False),
        ]
    )

    class EmbeddingModel:
        def predict(self, source, **kwargs):
            rows = [
                [{"path": [], "parents": [], "output_index": 0, "embedding": [float(value), 0.0, 0.0, 0.0]}]
                for value in source["id"].to_pylist()
            ]
            return pa.table({"representations": pa.array(rows, type=pa.list_(pa.field("element", representation)))})

    server = runtime(model=EmbeddingModel())

    outputs = server.predict_payloads([{"id": 2}, "invalid", {"id": 4}])

    assert "predictions" not in outputs[0]
    assert outputs[0]["representations"][0]["embedding"] == [2.0, 0.0, 0.0, 0.0]
    assert outputs[1]["predictions"] is None
    assert outputs[1]["error"]["status_code"] == 422
    assert outputs[2]["representations"][0]["embedding"] == [4.0, 0.0, 0.0, 0.0]


def test_runtime_keeps_user_postprocessor_columns_without_reserved_model_name():
    @rf.postprocess
    def rename(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(model=pl.col("predictions").struct.field("label"))

    assert runtime(postprocessor=rename).predict_payloads([{"id": 1}]) == [{"model": "1"}]


def test_runtime_preserves_request_fields_introduced_after_the_first_row():
    model = PredictModel()
    server = runtime(model=model)

    server.predict_payloads([{"id": 3}, {"id": 1, "context": "kept"}])

    assert model.calls[0]["source"].to_pylist() == [
        {"id": 3, "context": None},
        {"id": 1, "context": "kept"},
    ]


def test_runtime_preserves_request_order_around_validation_errors():
    class Request(pydantic.BaseModel):
        id: int = pydantic.Field(gt=0)

    model = PredictModel()
    server = runtime(model=model, request_signature=Request)

    outputs = server.predict_payloads([{"id": 2}, {"id": 0}, "wrong", {"id": 4}])

    assert len(model.calls) == 1
    assert model.calls[0]["source"].to_pylist() == [{"id": 2}, {"id": 4}]
    assert outputs[0] == {"predictions": {"label": "2"}}
    assert outputs[1]["error"]["status_code"] == 422
    assert "greater than 0" in outputs[1]["error"]["message"]
    assert outputs[2] == {
        "predictions": None,
        "error": {"status_code": 422, "message": "each request must be a JSON object, got str"},
    }
    assert outputs[3] == {"predictions": {"label": "4"}}


def test_runtime_uses_model_output_order_positionally():
    class ReorderingModel(PredictModel):
        def predict(self, source, *, preprocess, postprocess, retain, batch_size=None):
            reversed_source = source.take(pa.array([1, 0], type=pa.int64()))
            return super().predict(
                reversed_source,
                preprocess=preprocess,
                postprocess=postprocess,
                retain=retain,
                batch_size=batch_size,
            )

    server = runtime(model=ReorderingModel())

    assert server.predict_payloads([{"id": 3}, {"id": 9}]) == [
        {"predictions": {"label": "9"}},
        {"predictions": {"label": "3"}},
    ]


def test_response_preserves_explicitly_retained_inputs():
    server = runtime(retain=("id",))

    output = server.predict_payloads([{"id": 8}])[0]

    assert output == {"predictions": {"label": "8"}, "inputs": {"id": 8}}
    assert "identity" not in output


def test_postprocessors_run_inside_the_single_model_prediction_call():
    seen = []

    @rf.postprocess
    def compact(frame: pl.DataFrame) -> pl.DataFrame:
        seen.append("compact")
        return frame.select(
            request_id=pl.col("inputs").struct.field("id"),
            label=pl.col("predictions").struct.field("label"),
        )

    @rf.postprocess
    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        seen.append("finish")
        return frame.select("request_id", "label")

    model = PredictModel()
    server = runtime(model=model, postprocessor=(compact, finish), retain=("id",))

    output = server.predict_payloads([{"id": 5}])[0]

    assert len(model.calls) == 1
    pipeline = model.calls[0]["postprocess"]
    assert isinstance(pipeline, tuple)
    assert pipeline[:2] == (compact, finish)
    assert len(pipeline) == 2
    assert seen == ["compact", "finish"]
    assert output == {"request_id": 5, "label": "5"}


def test_deployment_postprocessor_must_preserve_request_cardinality():
    @rf.postprocess
    def discard(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.head(0)

    server = runtime(postprocessor=discard)

    with pytest.raises(ValueError, match="returned 0 rows for 1 valid"):
        server.predict_payloads([{"id": 1}])


def test_response_signature_validates_each_terminal_arrow_row():
    class Response(pydantic.BaseModel):
        predictions: dict[str, str]

    server = runtime(response_signature=Response)

    assert server.predict_payloads([{"id": 7}]) == [{"predictions": {"label": "7"}}]


def test_runtime_rejects_non_arrow_model_output():
    class BrokenModel:
        def predict(self, source, **kwargs):
            return [{"predictions": {}}]

    server = runtime(model=BrokenModel())

    with pytest.raises(TypeError, match="Model.predict must return a pyarrow.Table"):
        server.predict_payloads([{"id": 1}])


def test_runtime_rejects_prediction_row_count_drift():
    class BrokenModel:
        def predict(self, source, **kwargs):
            return pa.table({"predictions": pa.array([], type=pa.null())})

    server = runtime(model=BrokenModel())

    with pytest.raises(ValueError, match="returned 0 rows for 1 valid"):
        server.predict_payloads([{"id": 1}])


def test_runtime_reports_arrow_incompatible_requests_without_calling_model():
    model = PredictModel()
    server = runtime(model=model)
    opaque = object()

    output = server.predict_payloads([{"id": opaque}])[0]

    assert model.calls == []
    assert output["error"]["status_code"] == 422
    assert "Arrow-compatible" in output["error"]["message"]


def test_runtime_setup_applies_updates_before_device_placement(monkeypatch):
    calls = []

    class LoadedModel:
        def update(self, *predicates, **values):
            calls.append(("update", predicates, values))
            return self

        def warn_unbounded(self):
            calls.append(("warn",))

        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    loaded = LoadedModel()
    monkeypatch.setattr(deployment_module.Model, "load", classmethod(lambda cls, checkpoint: loaded))
    predicate = where("name") == "label"
    server = deployment_module.FastAPIRuntime(
        checkpoint="unused",
        accelerator=deployment_module.Accelerator.cpu,
        update_operations=[((predicate,), {"target": False})],
    )

    server.setup()

    assert calls == [
        ("update", (predicate,), {"target": False}),
        ("warn",),
        ("to", "cpu"),
        ("eval",),
    ]


def test_runtime_setup_uses_an_in_memory_model(monkeypatch):
    model = Model(amount=Number(), d_model=8, n_layers=1, n_heads=2)
    monkeypatch.setattr(
        deployment_module.Model,
        "load",
        classmethod(lambda cls, checkpoint: (_ for _ in ()).throw(AssertionError("checkpoint should not load"))),
    )
    server = deployment_module.FastAPIRuntime(
        checkpoint=model,
        accelerator=deployment_module.Accelerator.cpu,
    )

    server.setup()

    assert server.model is model


def test_deployment_builder_keeps_polars_processors_and_retention():
    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    @rf.postprocess
    def compact(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    deployment = Deployment(checkpoint="unused", retain=("request_id",)).preprocess(prepare).postprocess(compact)

    assert deployment.retain == ("request_id",)
    assert deployment._preprocessors == (prepare,)
    assert deployment._postprocessors == (compact,)


def test_deployment_builder_replaces_processor_collections():
    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    @rf.preprocess
    def replace(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    @rf.postprocess
    def compact(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    @rf.postprocess
    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    deployment = (
        Deployment(checkpoint="unused")
        .preprocess([prepare])
        .preprocess((replace,))
        .postprocess([compact])
        .postprocess((finish,))
    )

    assert deployment._preprocessors == (replace,)
    assert deployment._postprocessors == (finish,)


@pytest.mark.parametrize("retain", [("id", "id"), ("",), (1,), ([],), "id", ["id"]])
def test_deployment_rejects_invalid_retention(retain):
    with pytest.raises((TypeError, ValueError, pydantic.ValidationError)):
        Deployment(checkpoint="unused", retain=retain)


def test_deployment_forge_registers_openapi_signatures():
    class Request(pydantic.BaseModel):
        amount: float

    class Response(pydantic.BaseModel):
        predictions: dict[str, object]

    app = Deployment(checkpoint="unused").forge(request=Request, response=Response).app()
    schema = app.openapi()
    operation = schema["paths"]["/predict"]["post"]

    assert {"Request", "Response"} <= set(schema["components"]["schemas"])
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/Request"},
            {"type": "array", "items": {"$ref": "#/components/schemas/Request"}},
        ]
    }
    assert operation["summary"] == "Predict JSON or Arrow records"
    assert "same media type as the request" in operation["description"]
    assert operation["requestBody"]["content"]["application/vnd.apache.arrow.stream"] == {}
    assert operation["responses"]["200"]["content"]["application/vnd.apache.arrow.stream"] == {}
    assert {"200", "400", "415", "422"} <= set(operation["responses"])
    assert operation["responses"]["415"]["headers"]["Accept"]["example"] == (
        "application/json, application/vnd.apache.arrow.stream"
    )
    assert "DeploymentError" in schema["components"]["schemas"]
    assert "ModelProvenance" not in schema["components"]["schemas"]
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["anyOf"][0]["required"] == ["predictions"]
    assert "model" not in response_schema["anyOf"][0]["properties"]
    assert operation["responses"]["200"]["headers"]["X-RelFlow-Prediction-Format"]["example"] == "2"
    assert "X-RelFlow-Model-Version" in operation["responses"]["200"]["headers"]
    assert schema["components"]["schemas"]["DeploymentError"]["properties"]["predictions"] == {"type": "null"}


def test_runtime_headers_mirror_canonical_metadata_without_checkpoint_paths():
    server = runtime(model=SimpleNamespace(version="1.2.3"))

    assert server.headers() == {
        "X-RelFlow-Prediction-Format": "2",
        "X-RelFlow-Model-Version": "1.2.3",
    }


@pytest.mark.parametrize("empty", [False, True])
def test_arrow_deployment_batches_processed_rows_and_runs_global_processors_once(monkeypatch, empty):
    processed = []
    postprocessed = []
    batches = []

    @rf.preprocess(scope="dataset")
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        processed.append(frame.height)
        centered = frame.with_columns(value=pl.col("id") - pl.col("id").mean())
        return pl.concat([centered, centered])

    @rf.postprocess
    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        postprocessed.append(frame.height)
        return (
            frame.select(
                request_id=pl.col("inputs").struct.field("id"),
                value=pl.col("inputs").struct.field("value"),
            )
            .unique(subset="request_id", maintain_order=True)
            .with_row_index("row")
        )

    model = Model(value=Number, d_model=8, n_layers=1, n_heads=2, embed=True)
    hook = model.register_forward_pre_hook(lambda module, inputs: batches.append(inputs[0]["/value"].state.shape[0]))
    monkeypatch.setattr(
        deployment_module,
        "mappings",
        lambda *args, **kwargs: pytest.fail("Arrow deployment entered the Python mapping adapter"),
    )
    app = (
        Deployment(checkpoint=model, accelerator="cpu", max_batch_size=2, retain=("id", "value"))
        .preprocess(prepare)
        .postprocess(finish)
        .app()
    )
    source = pa.table({"id": [3, 1, 8]})
    if empty:
        source = source.slice(0, 0)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, source.schema) as writer:
        writer.write_table(source)

    class Request:
        headers = {"content-type": "application/vnd.apache.arrow.stream"}

        async def body(self):
            return sink.getvalue().to_pybytes()

    endpoint = next(route.endpoint for route in app.routes if route.path == "/predict")

    async def run():
        async with app.router.lifespan_context(app):
            return await endpoint(Request())

    try:
        response = asyncio.run(run())
    finally:
        hook.remove()

    assert response.status_code == 200
    assert response.media_type == "application/vnd.apache.arrow.stream"
    output = pa.ipc.open_stream(pa.BufferReader(response.body)).read_all()
    assert output.select(["row", "request_id", "value"]).to_pydict() == {
        "row": [] if empty else [0, 1, 2],
        "request_id": [] if empty else [3, 1, 8],
        "value": [] if empty else [-1.0, -3.0, 4.0],
    }
    assert "model" not in output.column_names
    assert response.headers["x-relflow-prediction-format"] == "2"
    assert response.headers["x-relflow-model-version"] == model.version
    assert processed == [source.num_rows]
    assert postprocessed == [2 * source.num_rows]
    assert batches == ([] if empty else [2, 2, 2])
    assert model.batch_size == 2


@pytest.mark.parametrize("stage", ["preprocess", "postprocess"])
def test_arrow_deployment_rejects_final_prediction_cardinality_drift(stage):
    @rf.preprocess
    def drop_inputs(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.head(2)

    @rf.postprocess
    def drop_outputs(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.head(2)

    model = Model(value=Number, d_model=8, n_layers=1, n_heads=2, embed=True)
    deployment = Deployment(checkpoint=model, accelerator="cpu", max_batch_size=2)
    if stage == "preprocess":
        deployment.preprocess(drop_inputs)
    else:
        deployment.postprocess(drop_outputs)
    app = deployment.app()
    source = pa.table({"value": [1.0, 2.0, 3.0]})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, source.schema) as writer:
        writer.write_table(source)

    class Request:
        headers = {"content-type": "application/vnd.apache.arrow.stream"}

        async def body(self):
            return sink.getvalue().to_pybytes()

    endpoint = next(route.endpoint for route in app.routes if route.path == "/predict")

    async def run():
        async with app.router.lifespan_context(app):
            return await endpoint(Request())

    with pytest.raises(ValueError, match="returned 2 rows for 3 valid deployment requests"):
        asyncio.run(run())


def test_deployment_predict_rejects_an_invalid_arrow_stream(monkeypatch):
    app = Deployment(checkpoint="unused", accelerator="cpu").app()

    class Request:
        headers = {"content-type": "application/vnd.apache.arrow.stream"}

        async def body(self):
            return b"not arrow"

    endpoint = next(route.endpoint for route in app.routes if route.path == "/predict")
    response = asyncio.run(endpoint(Request()))
    assert response.status_code == 400
    assert orjson.loads(response.body)["error"]["message"].startswith("invalid Arrow IPC stream:")


def test_deployment_predict_rejects_an_unsupported_content_type():
    app = Deployment(checkpoint="unused", accelerator="cpu").app()

    class Request:
        headers = {"content-type": "application/octet-stream"}

    endpoint = next(route.endpoint for route in app.routes if route.path == "/predict")
    response = asyncio.run(endpoint(Request()))

    assert response.status_code == 415
    assert response.headers["accept"] == "application/json, application/vnd.apache.arrow.stream"
    assert orjson.loads(response.body)["error"] == {
        "status_code": 415,
        "message": (
            "unsupported content type 'application/octet-stream'; use 'application/json' "
            "or 'application/vnd.apache.arrow.stream'"
        ),
    }
    assert all(route.path != "/predict-arrow" for route in app.routes)


def test_deployment_serve_configures_uvicorn(monkeypatch):
    captured = {}

    def run(app, *, host, port, log_level):
        captured.update(app=app, host=host, port=port, log_level=log_level)

    monkeypatch.setattr(deployment_module.uvicorn, "run", run)

    Deployment(
        checkpoint="unused",
        accelerator="cpu",
        host="127.0.0.1",
        port=8765,
        log_level="error",
    ).serve()

    assert isinstance(captured["app"], deployment_module.fastapi.FastAPI)
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["log_level"] == "error"


def test_multiworker_deployment_uses_the_importable_asgi_app(monkeypatch):
    captured = {}

    def run(app, *, workers, host, port, log_level):
        captured.update(app=app, workers=workers, host=host, port=port, log_level=log_level)

    monkeypatch.setattr(deployment_module.uvicorn, "run", run)

    Deployment(checkpoint="model.ckpt", workers=2).serve()

    assert captured["app"] == "relflow.inference.asgi:app"
    assert captured["workers"] == 2


def test_multiworker_deployment_rejects_in_process_builder_configuration():
    @rf.postprocess
    def compact(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    with pytest.raises(ValueError, match="cannot serialize in-process"):
        Deployment(checkpoint="model.ckpt", workers=2).postprocess(compact).serve()


def test_deployment_rejects_explicit_checkpoint_and_model():
    model = Model(amount=Number(), d_model=8, n_layers=1, n_heads=2)

    with pytest.raises(ValueError, match="pass either checkpoint or model"):
        Deployment(checkpoint="model.ckpt", model=model)


def test_error_item_is_stable():
    error = ErrorItem(status_code=422, message="bad input")

    assert runtime().failure(error) == {
        "predictions": None,
        "error": {"status_code": 422, "message": "bad input"},
    }
