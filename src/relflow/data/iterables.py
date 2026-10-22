"""Arrow batch encoding into policy-resolved tensorfields."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import torch
from tensordict import TensorDict

from relflow.data.arrow import Encoded, Geometry, Retain
from relflow.data.datasets.base import EncodedInput, InterprocessEncodingContext
from relflow.data.ragged import Layout, RaggedField, boolean, coalesce
from relflow.structs.enums import Component, Strata, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import TENSORFIELDS, BatchContext, Context, TensorFieldBase


def geometry(layouts: Mapping[Address, Layout], *, rows: int) -> TensorDict:
    """Move shared Arrow structural coordinates into the Torch batch plane."""

    axes: dict[Address, Geometry] = {}
    for address, layout in layouts.items():
        slots = np.zeros(layout.shape, dtype=bool)
        slots.reshape(-1)[layout.placement] = True
        axes[address] = Geometry(
            lengths=torch.from_numpy(layout.lengths),
            present=torch.from_numpy(layout.present),
            valid=torch.from_numpy(layout.valid),
            slots=torch.from_numpy(slots),
            batch_size=[rows],
        )
    return TensorDict(source=cast(Any, axes), batch_size=[rows])


def encode(
    batch: pa.Table,
    schema: Schema,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
    seed: int = 0,
    epoch: int = 0,
    retain: Retain = (),
) -> Encoded:
    out: dict[Address, TensorFieldBase] = {}
    observations: dict[Address, TensorDict] = {}
    bindings: dict[Address, object] = {}

    salt = random.getrandbits(64) if strata == Strata.train else 0

    coalesced = coalesce(batch, schema=schema, strata=strata, seed=seed, epoch=epoch)

    for address, request in schema.active_requests.items():
        extension = TENSORFIELDS[request.type]
        TensorField = extension.TensorField
        projection = coalesced.fields.pop(address)
        pristine = projection.pristine
        nulls = pristine.state.to_numpy(zero_copy_only=False) == Tokens.null.value
        if not request.nullable and nulls.any():
            raise ValueError(
                f"request '{address}' has nullable=False but input contains {int(nulls.sum())} null value(s)"
            )

        values = pa.nulls(0) if projection.vacant else extension.prepare(pristine.values, address=address)
        state = interprocess_encoding_context.get(address)
        if isinstance(state, BatchContext):
            if Component.bind not in extension.components:
                raise RuntimeError(f"extension '{extension.name}' at '{address}' requires a bind component")
            state, bindings[address] = state.batch(values)
        observation = None
        if strata == Strata.train:
            canonical = replace(pristine, values=values)
            observation = extension.observe(
                field=canonical,
                address=address,
                schema=schema,
                state=state,
                learn=True,
            )

        learner = extension.components.get(Component.learn)
        if observation is not None and not isinstance(observation, TensorDict):
            raise TypeError(
                f"extension '{extension.name}' observer at '{address}' must return a TensorDict or None, "
                f"got {type(observation).__name__}"
            )
        if observation is not None and learner is None:
            raise RuntimeError(
                f"extension '{extension.name}' observer at '{address}' returned an observation without a learner"
            )
        if strata == Strata.train and learner is not None and observation is None:
            raise RuntimeError(f"extension '{extension.name}' learner at '{address}' requires a training observation")
        if strata != Strata.train and observation is not None:
            raise RuntimeError(
                f"extension '{extension.name}' observer at '{address}' returned an observation with learn=False"
            )
        if observation is not None:
            observations[address] = observation

        input_field, target_field = projection.split(values)
        out[address] = TensorField.new(
            input=input_field,
            target=target_field,
            present=torch.from_numpy(boolean(projection.present).copy()).reshape(input_field.shape),
            trainable=torch.from_numpy(boolean(projection.trainable).copy()).reshape(input_field.shape),
            inferred=torch.from_numpy(boolean(projection.inferred).copy()).reshape(input_field.shape),
            address=address,
            schema=schema,
            strata=strata,
            context=Context(
                state=state,
                salt=salt,
            ),
        )

    inputs = TensorDict(source=cast(Any, out), batch_size=[batch.num_rows])
    inputs[Address()] = geometry(coalesced.layouts, rows=batch.num_rows)
    return Encoded(tensors=inputs, source=batch, retain=retain, observations=observations, bindings=bindings)


def mock(schema: Schema, batch_size: int) -> EncodedInput:
    out: dict[Address, TensorFieldBase] = {}
    axes: dict[Address, Geometry] = {}
    root = Address()
    axes[root] = Geometry(
        lengths=torch.ones(batch_size, dtype=torch.int64),
        present=torch.ones(batch_size, dtype=torch.bool),
        valid=torch.ones(batch_size, dtype=torch.bool),
        slots=torch.ones((batch_size, 1), dtype=torch.bool),
        batch_size=[batch_size],
    )
    required = {
        node.address for request in schema.active_requests.values() for node in request.path if node.type == "branch"
    }
    for address, branch in schema.branches.items():
        if address == root or address not in required:
            continue
        parent = axes[cast(Node, branch.parent).address].slots.shape
        axes[address] = Geometry(
            lengths=torch.zeros(parent, dtype=torch.int64),
            present=torch.zeros(parent, dtype=torch.bool),
            valid=torch.zeros(parent, dtype=torch.bool),
            slots=torch.zeros((*parent, 0), dtype=torch.bool),
            batch_size=[batch_size],
        )

    for address, request in schema.active_requests.items():
        TensorField = TENSORFIELDS[request.type].TensorField
        shape = tuple(axes[cast(Node, request.parent).address].slots.shape)
        state = pa.array(np.full(np.prod(shape), Tokens.padded.value, dtype=np.int8))
        field = RaggedField(
            values=pa.nulls(0),
            state=state,
            placement=pa.array([], type=pa.int64()),
            shape=shape,
        )
        routing = torch.zeros(shape, dtype=torch.bool)
        out[address] = TensorField.new(
            input=field,
            target=field,
            present=routing,
            trainable=routing,
            inferred=routing,
            address=address,
            schema=schema,
            strata=Strata.predict,
            context=Context(),
        )

    inputs = TensorDict(source=cast(Any, out), batch_size=[batch_size])
    inputs[root] = TensorDict(source=cast(Any, axes), batch_size=[batch_size])
    return inputs
