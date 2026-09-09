"""Polars preprocessor and postprocessor contracts."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from typing import Any, Literal, Self, TypeAlias, overload

import polars as pl
import pyarrow as pa

EMPTY = inspect.Signature.empty
Scope = Literal["partition", "dataset"]


class PreprocessorProvider(StrEnum):
    """Pipeline-owned preprocessor parameter names."""

    strata = "strata"
    schema = "schema"
    encoding_context = "encoding_context"


PREPROCESSOR_PROVIDERS = frozenset(provider.value for provider in PreprocessorProvider)


def label(func: Callable[..., Any]) -> str:
    """Return a stable display name for a processor callable."""

    return getattr(func, "__name__", type(func).__name__)


def frame(value: Any, *, kind: str, name: str) -> pl.DataFrame:
    """Validate one eager Polars processor result."""

    if isinstance(value, pl.LazyFrame):
        raise TypeError(f"{kind} '{name}' must return a polars.DataFrame, not LazyFrame")
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"{kind} '{name}' must return a polars.DataFrame, got {type(value).__name__}")
    if value.width == 0:
        raise ValueError(f"{kind} '{name}' must return at least one Polars column")
    return value


def polars(table: pa.Table, *, context: str) -> pl.DataFrame:
    """Convert one Arrow table to an eager Polars frame without row materialization."""

    if not isinstance(table, pa.Table):
        raise TypeError(f"{context} must be a pyarrow.Table, got {type(table).__name__}")
    if table.num_columns == 0:
        raise ValueError(f"{context} must contain at least one Arrow column")
    try:
        result = pl.from_arrow(table, rechunk=False)
    except Exception as error:
        raise TypeError(f"{context} could not be converted from Arrow to Polars: {error}") from error
    if not isinstance(result, pl.DataFrame):
        raise TypeError(f"{context} Arrow conversion returned {type(result).__name__}; expected DataFrame")
    return result


def arrow(value: pl.DataFrame, *, context: str) -> pa.Table:
    """Convert one eager Polars frame to the canonical Arrow representation."""

    frame(value, kind=context, name="output")
    try:
        return value.to_arrow(compat_level=pl.CompatLevel.oldest())
    except Exception as error:
        raise TypeError(f"{context} could not be converted from Polars to Arrow: {error}") from error


@dataclass(frozen=True, slots=True, kw_only=True)
class Processor:
    """Callable wrapper with explicit pipeline and user parameter binding."""

    func: Callable[..., Any]
    kind: str
    providers: frozenset[str]
    bound: Mapping[str, Any] = field(default_factory=dict)
    signature: inspect.Signature = field(init=False)
    runtime: frozenset[str] = field(init=False)
    user: frozenset[str] = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        signature = inspect.signature(self.func)
        runtime, user = self.classify(signature)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "runtime", frozenset(runtime))
        object.__setattr__(self, "user", frozenset(user))
        object.__setattr__(self, "name", label(self.func))
        self.bindings()

    def __reduce__(self):
        """Resolve decorated module functions by their public wrapper in workers."""

        configuration = {item.name: getattr(self, item.name) for item in fields(self) if item.init}
        reference = None
        if inspect.isfunction(self.func) and "<locals>" not in self.func.__qualname__:
            resolved = importlib.import_module(self.func.__module__)
            for name in self.func.__qualname__.split("."):
                resolved = getattr(resolved, name, None)
            if isinstance(resolved, Processor) and resolved.func is self.func:
                reference = (self.func.__module__, self.func.__qualname__)
                configuration.pop("func")
        return restore, (type(self), configuration, reference)

    @classmethod
    def normalize(cls, value: Self | list[Self] | tuple[Self, ...] | None) -> tuple[Self, ...]:
        """Normalize optional processor configuration to an immutable pipeline."""

        if value is None:
            return ()
        if isinstance(value, cls):
            processors = (value,)
        elif isinstance(value, (list, tuple)):
            processors = tuple(value)
        else:
            raise TypeError(
                f"{cls.__name__.lower()} must be a {cls.__name__}, list, tuple, or None; got {type(value).__name__}"
            )

        for index, processor in enumerate(processors):
            if not isinstance(processor, cls):
                raise TypeError(
                    f"{cls.__name__.lower()} at index {index} must be a {cls.__name__}; got {type(processor).__name__}"
                )
            processor.ready()
        return processors

    def classify(self, signature: inspect.Signature) -> tuple[set[str], set[str]]:
        """Classify callable parameters as pipeline-owned or user-bound."""

        parameters = list(signature.parameters.values())
        if not parameters:
            raise TypeError(f"{self.kind} '{label(self.func)}' must accept 'frame'")

        first = parameters[0]
        if first.name != "frame" or first.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(f"{self.kind} '{label(self.func)}' first parameter must be 'frame'")

        runtime: set[str] = set()
        user: set[str] = set()
        for parameter in parameters[1:]:
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(f"{self.kind} '{label(self.func)}' does not support *args")
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                raise TypeError(f"{self.kind} '{label(self.func)}' does not support **kwargs")
            if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(f"{self.kind} '{label(self.func)}' parameter '{parameter.name}' must be keyword-only")
            if parameter.name in self.providers:
                runtime.add(parameter.name)
            else:
                user.add(parameter.name)

        return runtime, user

    def bindings(self) -> None:
        """Validate explicitly bound user arguments."""

        for name in self.bound:
            if name in self.runtime:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is provided by the pipeline")
            if name not in self.user:
                raise ValueError(f"{self.kind} '{self.name}' has no user-bound parameter '{name}'")

    def ready(self) -> None:
        """Validate that every required user argument is bound."""

        self.bindings()
        missing = [
            name
            for name in sorted(self.user)
            if name not in self.bound and self.signature.parameters[name].default is EMPTY
        ]
        if missing:
            formatted = ", ".join(repr(name) for name in missing)
            raise ValueError(f"{self.kind} '{self.name}' requires unbound parameter(s): {formatted}")

    def partial(self, **values: Any) -> Self:
        """Return a processor with additional immutable user configuration."""

        for name in values:
            if name in self.runtime:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is provided by the pipeline")
            if name not in self.user:
                raise ValueError(f"{self.kind} '{self.name}' has no user-bound parameter '{name}'")
            if name in self.bound:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is already bound")

        return replace(self, bound={**self.bound, **values})

    def call(self, value: pl.DataFrame, runtime: Mapping[str, Any]) -> Any:
        """Invoke the wrapped callable with validated arguments."""

        if not isinstance(value, pl.DataFrame):
            raise TypeError(f"{self.kind} '{self.name}' requires a polars.DataFrame, got {type(value).__name__}")
        self.ready()
        missing = sorted(self.runtime - runtime.keys())
        if missing:
            formatted = ", ".join(repr(name) for name in missing)
            raise ValueError(f"{self.kind} '{self.name}' is missing pipeline parameter(s): {formatted}")
        supplied = {name: runtime[name] for name in self.runtime}
        return self.func(value, **dict(self.bound), **supplied)

    def __call__(self, frame: pl.DataFrame, **runtime: Any) -> Any:
        return self.call(frame, runtime)


@dataclass(frozen=True, slots=True, kw_only=True)
class Preprocessor(Processor):
    """Polars frame transform returned by :func:`preprocess`."""

    func: Callable[..., Any]
    scope: Scope = "partition"
    kind: str = field(default="preprocessor", init=False)
    providers: frozenset[str] = field(default=PREPROCESSOR_PROVIDERS, init=False)

    def __post_init__(self) -> None:
        if self.scope not in ("partition", "dataset"):
            raise ValueError("preprocessor scope must be 'partition' or 'dataset'")
        Processor.__post_init__(self)

    def run(
        self,
        value: pl.DataFrame,
        *,
        strata: Any,
        schema: Any,
        encoding_context: Any,
    ) -> Iterable[pl.DataFrame]:
        """Yield the zero, one, or many Polars frames produced by one call."""

        result = self.call(
            value,
            {
                PreprocessorProvider.strata.value: strata,
                PreprocessorProvider.schema.value: schema,
                PreprocessorProvider.encoding_context.value: encoding_context,
            },
        )
        if result is None:
            return
        if isinstance(result, pl.DataFrame):
            yield frame(result, kind=self.kind, name=self.name)
            return
        invalid = (str, bytes, Mapping, pl.Series, pl.LazyFrame, pa.Table, pa.RecordBatch)
        if isinstance(result, invalid) or not isinstance(result, Iterable):
            raise TypeError(
                f"preprocessor '{self.name}' must return DataFrame, Iterable[DataFrame], or None; "
                f"got {type(result).__name__}"
            )

        for item in result:
            if not isinstance(item, pl.DataFrame):
                raise TypeError(f"preprocessor '{self.name}' yielded {type(item).__name__}; expected DataFrame")
            yield frame(item, kind=self.kind, name=self.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class Postprocessor(Processor):
    """Polars output transform returned by :func:`postprocess`."""

    func: Callable[..., Any]
    kind: str = field(default="postprocessor", init=False)
    providers: frozenset[str] = field(default_factory=frozenset, init=False)

    def run(self, value: pl.DataFrame) -> pl.DataFrame:
        """Apply and validate one Polars output transform."""

        return frame(self.call(value, {}), kind=self.kind, name=self.name)


PreprocessorInput: TypeAlias = Preprocessor | list[Preprocessor] | tuple[Preprocessor, ...] | None
PostprocessorInput: TypeAlias = Postprocessor | list[Postprocessor] | tuple[Postprocessor, ...] | None


def restore(kind: type[Processor], configuration: dict[str, Any], reference: tuple[str, str] | None) -> Processor:
    """Rebuild processor configuration around an importable decorated function."""

    if reference is not None:
        module, qualified = reference
        resolved = importlib.import_module(module)
        for name in qualified.split("."):
            resolved = getattr(resolved, name)
        if not isinstance(resolved, Processor):
            raise TypeError(f"processor {module}.{qualified} must still refer to a decorated processor")
        configuration = {**configuration, "func": resolved.func}
    return kind(**configuration)


def apply(table: pa.Table, processors: PostprocessorInput = ()) -> pa.Table:
    """Apply an ordered Polars postprocessor pipeline to one Arrow table."""

    pipeline = Postprocessor.normalize(processors)
    if not pipeline:
        return table
    result = polars(table, context="postprocessor input")
    for processor in pipeline:
        result = processor.run(result)
    return arrow(result, context="postprocessor output")


@overload
def preprocess(
    func: Callable[..., Any],
    /,
    *,
    scope: Scope = "partition",
) -> Preprocessor: ...


@overload
def preprocess(
    func: None = None,
    /,
    *,
    scope: Scope = "partition",
) -> Callable[[Callable[..., Any]], Preprocessor]: ...


def preprocess(
    func: Callable[..., Any] | None = None,
    /,
    *,
    scope: Scope = "partition",
) -> Callable[[Callable[..., Any]], Preprocessor] | Preprocessor:
    """Wrap a callable as a Polars preprocessor."""

    def decorate(inner: Callable[..., Any]) -> Preprocessor:
        if not callable(inner):
            raise TypeError("preprocess can only decorate callables")
        return Preprocessor(func=inner, scope=scope)

    if func is None:
        return decorate
    return decorate(func)


@overload
def postprocess(func: Callable[..., Any], /) -> Postprocessor: ...


@overload
def postprocess(func: None = None, /) -> Callable[[Callable[..., Any]], Postprocessor]: ...


def postprocess(
    func: Callable[..., Any] | None = None,
    /,
) -> Callable[[Callable[..., Any]], Postprocessor] | Postprocessor:
    """Wrap a callable as a Polars postprocessor."""

    def decorate(inner: Callable[..., Any]) -> Postprocessor:
        if not callable(inner):
            raise TypeError("postprocess can only decorate callables")
        return Postprocessor(func=inner)

    if func is None:
        return decorate
    return decorate(func)


__all__ = [
    "Postprocessor",
    "Preprocessor",
    "PreprocessorProvider",
    "postprocess",
    "preprocess",
]
