"""Shared tensorfield construction for focused extension tests."""

from contextlib import nullcontext
from unittest.mock import patch

from relflow.data.ragged import Sources
from relflow.structs.experiment import Schema
from relflow.tensorfields.base import BatchContext, Context, TensorFieldBase


def tensorize(
    tensorfield: type[TensorFieldBase],
    sources: Sources,
    *,
    schema: Schema,
    context: Context = Context(),
) -> TensorFieldBase:
    """Exercise one field with the caller's vocabulary and salt."""
    state = context.state
    local = (
        patch.object(type(state), "batch", return_value=(state, None))
        if isinstance(state, BatchContext)
        else nullcontext()
    )
    with local:
        field, _, _ = tensorfield.new(sources, schema=schema, context=context)
    return field


__all__ = ["tensorize"]
