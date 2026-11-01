"""Realtime deployment over RelFlow's Arrow prediction boundary."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, cast

import fastapi
import orjson
import pyarrow as pa
import pyarrow.ipc as ipc
import pydantic
import torch
import uvicorn
from beartype import beartype
from fastapi.openapi.utils import get_openapi
from pydantic import AliasChoices, Field, field_validator
from pydantic_core import to_jsonable_python
from pydantic_settings import BaseSettings, SettingsConfigDict

from relflow.architecture.output import metadata
from relflow.architecture.root import Model
from relflow.data.arrow import Retain, mappings, retention
from relflow.data.processors import Postprocessor, PostprocessorInput, Preprocessor, PreprocessorInput
from relflow.logging import logger
from relflow.logging.config import LEVELS, configure
from relflow.structs.experiment import NodeAttribute, NodePredicate
from relflow.structs.tree import Node

Input: TypeAlias = dict[str, Any]
ModelSource: TypeAlias = str | Path | Model
UpdateOperation: TypeAlias = tuple[tuple[NodePredicate | NodeAttribute | Callable[[Node], bool], ...], dict[str, Any]]
DeploymentType = TypeVar("DeploymentType", bound="Deployment")
T = TypeVar("T")
JSON_MEDIA_TYPE = "application/json"
ARROW_MEDIA_TYPE = "application/vnd.apache.arrow.stream"
# Request content failures raised by Arrow conversion, preprocessing, and tensorfields.
DATA_ERRORS = (TypeError, ValueError, KeyError, pa.ArrowException)
RETRY_HEADERS = {"Retry-After": "1"}


class Accelerator(StrEnum):
    """Select automatic device discovery, CPU, CUDA or Apple MPS for serving."""

    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"

    @classmethod
    def _missing_(cls, value: object) -> Accelerator | None:
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()
        if normalized == "":
            raise ValueError("accelerator must not be blank")
        return cast(Accelerator | None, cls._value2member_map_.get(normalized))


class JSONBackend(StrEnum):
    """Choose the optional fast JSON encoder or Python's standard JSON encoder."""

    orjson = "orjson"
    stdlib = "stdlib"

    @classmethod
    def _missing_(cls, value: object) -> JSONBackend | None:
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()
        if normalized == "":
            raise ValueError("json_backend must not be blank")
        return cast(JSONBackend | None, cls._value2member_map_.get(normalized))


class ErrorItem(pydantic.BaseModel):
    """One request-local validation error."""

    status_code: int
    message: str


class Unavailable(RuntimeError):
    """The deployment cannot take work right now; the client should retry."""


def envelope(status_code: int, message: str) -> dict[str, Any]:
    """Build the stable error envelope shared by request-local and transport-level failures."""

    return {"predictions": None, "error": {"status_code": status_code, "message": message}}


def records(rows: Sequence[Input]) -> pa.Table:
    """Cross the Python-to-Arrow boundary once for one group of requests."""

    try:
        return pa.Table.from_pylist(mappings(rows, context="deployment requests"))
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise ValueError(f"request is not Arrow-compatible: {error}") from error


def jsonable(rows: list[Any]) -> list[Any]:
    """Settle the JSON representation once for both encoders.

    Dates, times, and durations use ISO 8601, binary uses URL-safe base64,
    decimals become strings, and NaN or infinite numbers become null.
    """

    return to_jsonable_python(rows, timedelta_mode="iso8601", bytes_mode="base64", inf_nan_mode="null")


class FastAPIRuntime:
    """Load one model and predict Arrow inputs for HTTP requests."""

    def __init__(
        self,
        *,
        checkpoint: ModelSource,
        accelerator: Accelerator,
        preprocessor: PreprocessorInput = (),
        postprocessor: PostprocessorInput = (),
        retain: Retain = (),
        update_operations: list[UpdateOperation] | None = None,
        request_signature: type[pydantic.BaseModel] | None = None,
        response_signature: type[pydantic.BaseModel] | None = None,
    ) -> None:
        self.model_source = checkpoint
        self.accelerator = accelerator
        self.preprocessors = Preprocessor.normalize(preprocessor)
        self.postprocessors = Postprocessor.normalize(postprocessor)
        self.retain = retention(retain)
        self.update_operations = list(update_operations or [])
        self.request_signature = request_signature
        self.response_signature = response_signature
        self.responses: pydantic.TypeAdapter[list[pydantic.BaseModel]] | None = (
            None if response_signature is None else pydantic.TypeAdapter(list[response_signature])
        )
        self.device = torch.device("cpu")

    def setup(self) -> None:
        """Load, mutate, place, and freeze the configured model."""

        if self.accelerator == Accelerator.auto:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                mps = getattr(torch.backends, "mps", None)
                self.device = torch.device("mps" if mps is not None and mps.is_available() else "cpu")
        else:
            self.device = torch.device(self.accelerator.value)

        model = self.model_source if isinstance(self.model_source, Model) else Model.load(self.model_source)
        for predicates, values in self.update_operations:
            model.update(*predicates, **values)
        model.warn_unbounded()

        self.model = model.to(self.device)
        self.model.eval()

    def validate(self, payload: Any) -> Input | ErrorItem:
        """Validate one JSON object without changing its request position."""

        if not isinstance(payload, dict):
            return ErrorItem(
                status_code=422,
                message=f"each request must be a JSON object, got {type(payload).__name__}",
            )

        try:
            if self.request_signature is None:
                return cast(Input, payload)
            request = self.request_signature.model_validate(payload)
            return request.model_dump(mode="python")
        except Exception as exception:
            return ErrorItem(status_code=422, message=str(exception))

    def headers(self) -> dict[str, str]:
        """Mirror the canonical prediction metadata without repeating it in rows."""

        values = metadata(self.model)
        return {
            "X-RelFlow-Model-Version": values[b"relflow.model_version"].decode("utf-8"),
        }

    def predict_table(self, data: pa.Table, *, batch_size: int | None = None) -> pa.Table:
        """Predict Arrow inputs without changing the selected output columns.

        Serving processors must preserve request identity and order, with one
        final output row per valid request. The row-count guard detects changed
        cardinality; processors remain responsible for identity and order. A
        supplied ``batch_size`` bounds encoding without changing the model's
        own batch size.
        """

        if not isinstance(data, pa.Table):
            raise TypeError(f"deployment prediction input must be a pyarrow.Table, got {type(data).__name__}")

        model = self.model
        previous = None if batch_size is None else model.batch_size
        try:
            result = model.predict(
                data,
                preprocess=self.preprocessors,
                postprocess=self.postprocessors,
                retain=self.retain,
                batch_size=batch_size,
            )
        finally:
            # ``Model.predict`` adopts a supplied size as the model's current batch size.
            if previous is not None:
                model.batch_size = previous
        if not isinstance(result, pa.Table):
            raise RuntimeError(f"Model.predict must return a pyarrow.Table, got {type(result).__name__}")
        if result.num_rows != data.num_rows:
            raise RuntimeError(
                f"Model.predict returned {result.num_rows} rows for {data.num_rows} valid deployment requests"
            )
        return result

    def predict_rows(self, rows: list[Input]) -> list[dict[str, Any]]:
        """Predict one group of valid requests together.

        A data error in a group of several requests bisects the group so only
        the offending requests receive a 422 envelope; processors rerun on the
        subsets. Runtime contract violations propagate unchanged.
        """

        try:
            result = self.predict_table(records(rows))
        except DATA_ERRORS as error:
            if len(rows) > 1:
                half = len(rows) // 2
                return self.predict_rows(rows[:half]) + self.predict_rows(rows[half:])
            return [envelope(422, str(error))]
        return self.serialize(result)

    def serialize(self, result: pa.Table) -> list[dict[str, Any]]:
        """Materialize JSON rows once per batch, through the response signature when one is attached."""

        rows = result.to_pylist()
        if self.responses is not None:
            rows = self.responses.dump_python(self.responses.validate_python(rows), mode="json")
        return jsonable(rows)

    def predict_payloads(self, payloads: list[Any]) -> list[dict[str, Any]]:
        """Validate, predict, and serialize one request microbatch, keeping request slots.

        Valid requests cross the Python-to-Arrow boundary together, so ordered
        preprocessing stays batchwise and ``Model.predict`` runs once for the
        group. Invalid request rows retain their original slots.
        """

        outputs: list[dict[str, Any] | None] = [None] * len(payloads)
        valid: list[Input] = []
        positions: list[int] = []

        for position, payload in enumerate(payloads):
            item = self.validate(payload)
            if isinstance(item, ErrorItem):
                outputs[position] = envelope(item.status_code, item.message)
            else:
                valid.append(item)
                positions.append(position)

        if valid:
            for position, output in zip(positions, self.predict_rows(valid), strict=True):
                outputs[position] = output

        return [cast(dict[str, Any], output) for output in outputs]


@dataclass(slots=True)
class QueuedRequest:
    payload: Any
    future: asyncio.Future[dict[str, Any]]


class FastAPIBatcher:
    """Gather concurrent HTTP requests into bounded model microbatches.

    One lock serializes every model call, so JSON microbatches and Arrow
    tables never run ``Model.predict`` concurrently. The queue admits at most
    ``max_queue_size`` pending requests; clients beyond it receive 503 instead
    of waiting without bound.
    """

    def __init__(
        self,
        runtime: FastAPIRuntime,
        *,
        max_batch_size: int,
        batch_timeout: float,
        max_queue_size: int = 1024,
    ) -> None:
        self.runtime = runtime
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=max_queue_size)
        self.lock = asyncio.Lock()
        self.task: asyncio.Task[None] | None = None

    @property
    def ready(self) -> bool:
        return self.task is not None and not self.task.done()

    async def start(self) -> None:
        self.runtime.setup()
        self.task = asyncio.create_task(self.work())

    async def stop(self) -> None:
        if self.task is None:
            return

        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task
        self.task = None

    async def run(self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run one model call on a worker thread while holding the model lock."""

        if not self.ready:
            raise Unavailable("deployment worker is not running; restart the server")
        async with self.lock:
            return await asyncio.to_thread(function, *args, **kwargs)

    async def submit(self, payload: Any) -> dict[str, Any]:
        return (await self.submit_many([payload]))[0]

    async def submit_many(self, payloads: list[Any]) -> list[dict[str, Any]]:
        if not payloads:
            return []
        if not self.ready:
            raise Unavailable("deployment worker is not running; restart the server")
        pending, capacity = self.queue.qsize(), self.queue.maxsize
        if pending + len(payloads) > capacity:
            raise Unavailable(
                f"deployment queue cannot take {len(payloads)} more requests with {pending} of {capacity} pending; "
                "retry later, send fewer records per request, or raise max_queue_size"
            )

        loop = asyncio.get_running_loop()
        items = [QueuedRequest(payload=payload, future=loop.create_future()) for payload in payloads]
        for item in items:
            self.queue.put_nowait(item)
        return list(await asyncio.gather(*(item.future for item in items)))

    async def work(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            self.drain(batch)

            if len(batch) < self.max_batch_size and self.batch_timeout > 0.0:
                deadline = asyncio.get_running_loop().time() + self.batch_timeout
                while len(batch) < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0.0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                    except TimeoutError:
                        break
                    self.drain(batch)

            payloads = [item.payload for item in batch]
            try:
                async with self.lock:
                    responses = await asyncio.to_thread(self.runtime.predict_payloads, payloads)
                if len(responses) != len(batch):
                    raise RuntimeError(
                        f"deployment runtime returned {len(responses)} responses for {len(batch)} requests"
                    )
            except Exception as exception:
                logger.bind(component="deployment", requests=len(batch)).exception("prediction batch failed")
                responses = [envelope(500, f"{type(exception).__name__}: {exception}")] * len(batch)

            for item, response in zip(batch, responses, strict=True):
                if not item.future.cancelled():
                    item.future.set_result(response)

    def drain(self, batch: list[QueuedRequest]) -> None:
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return


class Deployment(BaseSettings):
    """Serving configuration for a checkpoint or an in-memory model.

    Pydantic request signatures validate each item before the valid rows are
    converted into one Arrow table. ``retain`` controls which processed input
    columns appear in the canonical output and are available to a Polars
    postprocessor. Model provenance is carried in Arrow metadata and HTTP
    headers. Processors must preserve request identity, order and cardinality.
    JSON responses represent dates, times, and durations as ISO 8601 strings,
    binary as URL-safe base64, decimals as strings, and NaN or infinite
    numbers as null, whichever ``json_backend`` encodes them.

    Set ``checkpoint`` to a path or provide an in-memory ``model``. Configure
    ``max_batch_size`` and ``batch_timeout`` for microbatching and
    ``max_queue_size`` for the pending-request bound beyond which clients
    receive 503; ``workers`` selects server processes. ``accelerator`` chooses
    the model device. ``host``, ``port`` and ``log_level`` configure Uvicorn,
    and ``serve`` applies ``log_level`` to RelFlow's own logger. Values can
    also come from the documented ``RELFLOW_*`` environment variables, where
    ``RELFLOW_RETAIN`` holds JSON such as ``'["amount"]'`` or ``'"*"'``.
    Configuration methods mutate this object and return it, preserving
    subclass methods during chaining.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        validate_by_name=True,
        validate_by_alias=True,
        arbitrary_types_allowed=True,
    )

    checkpoint: ModelSource = Field(
        default="model.ckpt",
        validation_alias=AliasChoices("RELFLOW_CHECKPOINT", "CHECKPOINT"),
    )
    model: Model | None = Field(default=None, exclude=True)
    max_batch_size: int = Field(
        default=128,
        ge=1,
        validation_alias=AliasChoices("RELFLOW_MAX_BATCH_SIZE", "MAX_BATCH_SIZE"),
    )
    batch_timeout: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("RELFLOW_BATCH_TIMEOUT", "BATCH_TIMEOUT"),
    )
    max_queue_size: int = Field(
        default=1024,
        ge=1,
        validation_alias=AliasChoices("RELFLOW_MAX_QUEUE_SIZE", "MAX_QUEUE_SIZE"),
    )
    workers: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("RELFLOW_WORKERS", "WORKERS"),
    )
    accelerator: Accelerator = Field(
        default=Accelerator.auto,
        validation_alias=AliasChoices("RELFLOW_ACCELERATOR", "ACCELERATOR"),
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("RELFLOW_HOST", "HOST"),
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("RELFLOW_PORT", "PORT"),
    )
    log_level: str = Field(
        default="info",
        validation_alias=AliasChoices("RELFLOW_LOG_LEVEL", "LOG_LEVEL"),
    )
    json_backend: JSONBackend = Field(
        default=JSONBackend.orjson,
        validation_alias=AliasChoices("RELFLOW_JSON_BACKEND", "JSON_BACKEND"),
    )
    retain: Retain = Field(
        default=(),
        validation_alias=AliasChoices("RELFLOW_RETAIN", "RETAIN"),
    )

    _request_signature: type[pydantic.BaseModel] | None = pydantic.PrivateAttr(default=None)
    _response_signature: type[pydantic.BaseModel] | None = pydantic.PrivateAttr(default=None)
    _preprocessors: tuple[Preprocessor, ...] = pydantic.PrivateAttr(default=())
    _postprocessors: tuple[Postprocessor, ...] = pydantic.PrivateAttr(default=())
    _update_operations: list[UpdateOperation] = pydantic.PrivateAttr(default_factory=list)

    def __init__(self, /, **values: Any) -> None:
        super().__init__(**values)
        # An environment checkpoint is a default; only an explicit one conflicts with an in-memory model.
        if self.model is not None and {"checkpoint", "RELFLOW_CHECKPOINT", "CHECKPOINT"} & values.keys():
            raise ValueError("pass either checkpoint or model, not both")

    @field_validator("checkpoint", mode="before")
    @classmethod
    def check_checkpoint(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                raise ValueError("checkpoint must not be blank")
            return stripped
        return value

    @field_validator("retain", mode="before")
    @classmethod
    def check_retain(cls, value: Any) -> Retain:
        names = tuple(value) if isinstance(value, list) else value
        try:
            return retention(cast(Retain, names))
        except TypeError as error:
            raise ValueError(str(error)) from error

    @field_validator("log_level", mode="before")
    @classmethod
    def check_log_level(cls, value: Any) -> str:
        if not isinstance(value, str) or value.strip().lower() not in LEVELS:
            raise ValueError(f"log_level must be one of {', '.join(LEVELS)}, got {value!r}")
        return value.strip().lower()

    @beartype
    def forge(
        self: DeploymentType,
        request: type[pydantic.BaseModel] | None = None,
        response: type[pydantic.BaseModel] | None = None,
    ) -> DeploymentType:
        """Attach optional Pydantic request and response signatures."""

        self._request_signature = request
        self._response_signature = response
        return self

    @beartype
    def preprocess(self: DeploymentType, preprocessor: PreprocessorInput) -> DeploymentType:
        """Attach an ordered eager Polars preprocessor collection."""

        self._preprocessors = Preprocessor.normalize(preprocessor)
        return self

    @beartype
    def postprocess(self: DeploymentType, postprocessor: PostprocessorInput) -> DeploymentType:
        """Attach an ordered eager Polars postprocessor collection."""

        self._postprocessors = Postprocessor.normalize(postprocessor)
        return self

    @beartype
    def update(
        self: DeploymentType,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> DeploymentType:
        """Queue declared schema changes for startup; use ``description`` for notes."""

        self._update_operations.append(
            (
                tuple(predicates),
                {
                    "strict": strict,
                    "include_root": include_root,
                    "validate": validate,
                    **values,
                },
            )
        )
        return self

    def decode(self, body: bytes) -> Any:
        """Parse one JSON request body with the configured backend."""

        if self.json_backend == JSONBackend.orjson:
            return orjson.loads(body)
        return json.loads(body)

    def encode(self, content: Any) -> bytes:
        """Serialize settled JSON content with the configured backend."""

        if self.json_backend == JSONBackend.orjson:
            return orjson.dumps(content)
        return json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def app(self) -> fastapi.FastAPI:
        """Build a FastAPI app for the configured model."""

        runtime = FastAPIRuntime(
            checkpoint=self.model if self.model is not None else self.checkpoint,
            accelerator=self.accelerator,
            preprocessor=self._preprocessors,
            postprocessor=self._postprocessors,
            retain=self.retain,
            update_operations=self._update_operations,
            request_signature=self._request_signature,
            response_signature=self._response_signature,
        )
        batcher = FastAPIBatcher(
            runtime=runtime,
            max_batch_size=self.max_batch_size,
            batch_timeout=self.batch_timeout,
            max_queue_size=self.max_queue_size,
        )

        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            await batcher.start()
            app.state.relflow_runtime = runtime
            app.state.relflow_batcher = batcher
            try:
                yield
            finally:
                await batcher.stop()

        app = fastapi.FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        def reply(
            content: Any,
            status_code: int = 200,
            headers: dict[str, str] | None = None,
        ) -> fastapi.Response:
            return fastapi.Response(
                content=self.encode(content),
                status_code=status_code,
                media_type=JSON_MEDIA_TYPE,
                headers=headers,
            )

        @app.post("/predict")
        async def predict(request: fastapi.Request) -> fastapi.Response:
            media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
            if media_type not in (JSON_MEDIA_TYPE, ARROW_MEDIA_TYPE):
                return reply(
                    envelope(
                        415,
                        f"unsupported content type {media_type or '<missing>'!r}; "
                        f"use {JSON_MEDIA_TYPE!r} or {ARROW_MEDIA_TYPE!r}",
                    ),
                    status_code=415,
                    headers={"Accept": f"{JSON_MEDIA_TYPE}, {ARROW_MEDIA_TYPE}"},
                )

            body = await request.body()
            if media_type == ARROW_MEDIA_TYPE:
                try:
                    table = ipc.open_stream(pa.BufferReader(body)).read_all()
                except (pa.ArrowException, OSError, ValueError) as exception:
                    return reply(envelope(400, f"invalid Arrow IPC stream: {exception}"), status_code=400)

                try:
                    predicted = await batcher.run(runtime.predict_table, table, batch_size=self.max_batch_size)
                except Unavailable as exception:
                    return reply(envelope(503, str(exception)), status_code=503, headers=RETRY_HEADERS)
                except DATA_ERRORS as exception:
                    return reply(envelope(422, str(exception)), status_code=422, headers=runtime.headers())
                except Exception as exception:
                    logger.bind(component="deployment", rows=table.num_rows).exception("Arrow prediction failed")
                    return reply(
                        envelope(500, f"{type(exception).__name__}: {exception}"),
                        status_code=500,
                        headers=runtime.headers(),
                    )
                sink = pa.BufferOutputStream()
                with ipc.new_stream(sink, predicted.schema) as writer:
                    writer.write_table(predicted)
                return fastapi.Response(
                    content=memoryview(sink.getvalue()),
                    media_type=ARROW_MEDIA_TYPE,
                    headers=runtime.headers(),
                )

            try:
                payload = self.decode(body)
            except ValueError as exception:
                return reply(envelope(400, str(exception)), status_code=400)

            if isinstance(payload, dict):
                payloads = [payload]
            elif isinstance(payload, list):
                payloads = payload
            else:
                return reply(
                    envelope(
                        422,
                        f"request body must be a JSON object or an array of JSON objects, got {type(payload).__name__}",
                    ),
                    status_code=422,
                )

            try:
                responses = await batcher.submit_many(payloads)
            except Unavailable as exception:
                return reply(envelope(503, str(exception)), status_code=503, headers=RETRY_HEADERS)
            content = responses if isinstance(payload, list) else responses[0]
            return reply(content, headers=runtime.headers())

        self.register(app)
        return app

    def register(self, app: fastapi.FastAPI) -> None:
        """Register both wire formats and optional Pydantic signatures in OpenAPI."""

        def register_model(openapi_schema: dict[str, Any], model: type[pydantic.BaseModel]) -> str:
            schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
            definitions = schema.pop("$defs", {})
            components = openapi_schema.setdefault("components", {}).setdefault("schemas", {})
            for name, definition in definitions.items():
                components.setdefault(name, definition)
            name = str(schema.get("title") or model.__name__)
            components.setdefault(name, schema)
            return name

        def cardinality(item: dict[str, Any]) -> dict[str, Any]:
            return {"anyOf": [item, {"type": "array", "items": item}]}

        def schema() -> dict[str, Any]:
            if app.openapi_schema is not None:
                return app.openapi_schema

            openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                summary=app.summary,
                description=app.description,
                routes=app.routes,
            )
            operation = openapi_schema["paths"]["/predict"]["post"]
            operation["summary"] = "Predict JSON or Arrow records"
            operation["description"] = (
                f"Set `Content-Type` to `{JSON_MEDIA_TYPE}` for JSON or `{ARROW_MEDIA_TYPE}` for an Arrow IPC "
                "stream. A successful response uses the same media type as the request. JSON request and response "
                "signatures apply only to the JSON representation; Arrow schemas are validated by preprocessing "
                "and the model. The response representation is not independently negotiated with `Accept`."
            )
            generic_request = {
                "anyOf": [
                    {"type": "object"},
                    {"type": "array", "items": {"type": "object"}},
                ]
            }
            if self._request_signature is not None:
                request_name = register_model(openapi_schema, self._request_signature)
                request_schema = cardinality({"$ref": f"#/components/schemas/{request_name}"})
            else:
                request_schema = generic_request
            operation["requestBody"] = {
                "required": True,
                "description": (
                    "One JSON record, an array of JSON records, or one Arrow IPC stream. "
                    "The request Content-Type selects the deserializer."
                ),
                "content": {
                    JSON_MEDIA_TYPE: {"schema": request_schema},
                    ARROW_MEDIA_TYPE: {},
                },
            }
            success = operation.setdefault("responses", {}).setdefault("200", {})
            success["description"] = (
                "Prediction output in the request's media type. JSON output preserves object/array cardinality, "
                "reports request-local failures as error envelopes in their slots, and represents dates, times, "
                "and durations as ISO 8601 strings, binary as URL-safe base64, decimals as strings, and NaN or "
                "infinite numbers as null; Arrow output is an IPC stream whose rows remain in input order. "
                "Selected canonical columns are preserved. Model provenance is carried in an HTTP header "
                "and Arrow schema metadata."
            )
            success["headers"] = {
                "X-RelFlow-Model-Version": {
                    "description": (
                        "RelFlow version stored in the loaded model. This describes the package/checkpoint "
                        "version and is not a unique weights fingerprint."
                    ),
                    "schema": {"type": "string"},
                },
            }
            responses = success.setdefault("content", {})
            components = openapi_schema.setdefault("components", {}).setdefault("schemas", {})
            if self._response_signature is not None:
                response_name = register_model(openapi_schema, self._response_signature)
                item = components[response_name]
            else:
                item = {"type": "object"}
            response_schema = cardinality(item)
            responses[JSON_MEDIA_TYPE] = {"schema": response_schema}
            responses[ARROW_MEDIA_TYPE] = {}

            components.setdefault(
                "DeploymentError",
                {
                    "type": "object",
                    "required": ["predictions", "error"],
                    "properties": {
                        "predictions": {"type": "null"},
                        "error": {
                            "type": "object",
                            "required": ["status_code", "message"],
                            "properties": {
                                "status_code": {"type": "integer"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                },
            )
            error = {"schema": {"$ref": "#/components/schemas/DeploymentError"}}
            operation["responses"]["400"] = {
                "description": "The declared JSON or Arrow IPC representation is malformed.",
                "content": {JSON_MEDIA_TYPE: error},
            }
            operation["responses"]["415"] = {
                "description": "The request Content-Type is missing or unsupported.",
                "headers": {
                    "Accept": {
                        "description": "Media types accepted as request content.",
                        "schema": {"type": "string"},
                        "example": f"{JSON_MEDIA_TYPE}, {ARROW_MEDIA_TYPE}",
                    }
                },
                "content": {JSON_MEDIA_TYPE: error},
            }
            operation["responses"]["422"] = {
                "description": (
                    "The JSON body is not an object or an array of objects, or the Arrow input cannot be "
                    "predicted; the envelope names the offending field."
                ),
                "content": {JSON_MEDIA_TYPE: error},
            }
            operation["responses"]["500"] = {
                "description": "Prediction failed inside the deployment; the envelope names the exception.",
                "content": {JSON_MEDIA_TYPE: error},
            }
            operation["responses"]["503"] = {
                "description": "The deployment worker is not running or its queue is full; retry after `Retry-After`.",
                "headers": {
                    "Retry-After": {
                        "description": "Seconds to wait before retrying.",
                        "schema": {"type": "string"},
                        "example": "1",
                    }
                },
                "content": {JSON_MEDIA_TYPE: error},
            }

            app.openapi_schema = openapi_schema
            return openapi_schema

        app.openapi = schema  # type: ignore[method-assign]

    def serve(self) -> None:
        """Start the configured FastAPI server with RelFlow logging at the same level."""

        configure(level=self.log_level)
        if self.workers > 1:
            if self.model is not None or isinstance(self.checkpoint, Model):
                raise ValueError("workers > 1 requires a checkpoint path, not an in-memory model")
            if any(
                (
                    self._request_signature is not None,
                    self._response_signature is not None,
                    bool(self._preprocessors),
                    bool(self._postprocessors),
                    bool(self._update_operations),
                )
            ):
                raise ValueError(
                    "workers > 1 cannot serialize in-process forge, preprocess, postprocess, or update configuration"
                )

            env_updates = {
                "RELFLOW_CHECKPOINT": str(self.checkpoint),
                "RELFLOW_MAX_BATCH_SIZE": str(self.max_batch_size),
                "RELFLOW_BATCH_TIMEOUT": str(self.batch_timeout),
                "RELFLOW_MAX_QUEUE_SIZE": str(self.max_queue_size),
                "RELFLOW_ACCELERATOR": self.accelerator.value,
                "RELFLOW_JSON_BACKEND": self.json_backend.value,
                "RELFLOW_LOG_LEVEL": self.log_level,
                "RELFLOW_RETAIN": orjson.dumps(self.retain).decode(),
            }
            previous = {key: os.environ.get(key) for key in env_updates}
            try:
                os.environ.update(env_updates)
                uvicorn.run(
                    "relflow.inference.asgi:app",
                    workers=self.workers,
                    host=self.host,
                    port=self.port,
                    log_level=self.log_level,
                )
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            return

        uvicorn.run(self.app(), host=self.host, port=self.port, log_level=self.log_level)


__all__ = [
    "Accelerator",
    "Deployment",
    "FastAPIBatcher",
    "FastAPIRuntime",
    "Input",
    "JSONBackend",
    "ModelSource",
    "Retain",
    "Unavailable",
    "UpdateOperation",
]
