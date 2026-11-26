"""Arrow batch encoding into policy-resolved tensorfields."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import numpy as np
import pyarrow as pa
import torch
from tensordict import TensorDict

from relflow.data.arrow import Encoded, Geometry, Retain
from relflow.data.datasets.base import EncodedInput, InterprocessEncodingContext
from relflow.data.ragged import Coalesced, Sources, coalesce
from relflow.structs.enums import Component, Strata
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address, Node
from relflow.structs.twigs import indexed
from relflow.tensorfields.base import TENSORFIELDS, Context, TensorFieldBase


def geometry(coalesced: Coalesced, *, rows: int) -> TensorDict:
    """Move shared Arrow structural coordinates and index positions into the Torch batch plane."""

    positions = coalesced.positions
    axes: dict[Address, Geometry] = {}
    for address, layout in coalesced.layouts.items():
        slots = np.zeros(layout.shape, dtype=bool)
        slots.reshape(-1)[layout.placement] = True
        axes[address] = Geometry(
            lengths=torch.from_numpy(layout.lengths),
            present=torch.from_numpy(layout.present),
            valid=torch.from_numpy(layout.valid),
            slots=torch.from_numpy(slots),
            positions=torch.from_numpy(positions[address]) if address in positions else None,
            batch_size=[rows],
        )
    return TensorDict(source=cast(Any, axes), batch_size=[rows])


def batch_salt(seed: int, *, epoch: int, ordinal: int) -> int:
    """The hash salt of one training batch.

    Hash fields are salted per batch so no bucket identity survives across
    batches, while every field in the batch shares the salt and keeps equality.
    Deriving it from the run seed, epoch, and batch ordinal keeps a seeded run
    reproducible; the process-wide ``random`` state is not, because other
    callers consume it.
    """

    payload = f"{seed}:{epoch}:{ordinal}:salt".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def encode(
    batch: pa.Table,
    schema: Schema,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
    seed: int = 0,
    epoch: int = 0,
    retain: Retain = (),
    ordinal: int = 0,
) -> Encoded:
    out: dict[Address, TensorFieldBase] = {}
    observations: dict[Address, TensorDict] = {}
    bindings: dict[Address, object] = {}

    salt = batch_salt(seed, epoch=epoch, ordinal=ordinal) if strata == Strata.train else 0

    coalesced = coalesce(batch, schema=schema, strata=strata, seed=seed, epoch=epoch, ordinal=ordinal)

    for address, request in schema.active_requests.items():
        extension = TENSORFIELDS[request.type]
        TensorField = extension.TensorField
        sources = coalesced.fields.pop(address)
        field, observation, binding = TensorField.new(
            sources,
            schema=schema,
            context=Context(state=interprocess_encoding_context.get(address), salt=salt),
        )
        if not isinstance(field, TensorField):
            raise TypeError(
                f"extension '{extension.name}' tensorfield at '{address}' must return {TensorField.__name__}, "
                f"got {type(field).__name__}"
            )
        if observation is not None and not isinstance(observation, TensorDict):
            raise TypeError(
                f"extension '{extension.name}' observer at '{address}' must return a TensorDict or None, "
                f"got {type(observation).__name__}"
            )
        learner = extension.components.get(Component.learn)
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
        if not isinstance(binding, tuple) or len(binding) > 1:
            raise TypeError(f"extension '{extension.name}' tensorfield at '{address}' must return zero or one binding")
        if binding and Component.bind not in extension.components:
            raise RuntimeError(f"extension '{extension.name}' at '{address}' requires a bind component")
        out[address] = field
        if observation is not None:
            observations[address] = observation
        if binding:
            bindings[address] = binding[0]

    inputs = TensorDict(source=cast(Any, out), batch_size=[batch.num_rows])
    inputs[Address()] = geometry(coalesced, rows=batch.num_rows)
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
            positions=None if indexed(branch) is None else torch.zeros((*parent, 0), dtype=torch.float32),
            batch_size=[batch_size],
        )

    for address, request in schema.active_requests.items():
        TensorField = TENSORFIELDS[request.type].TensorField
        shape = tuple(axes[cast(Node, request.parent).address].slots.shape)
        sources = Sources(
            inputs=(),
            placement=np.array([], dtype=np.int64),
            shape=shape,
            decisions=(),
            vacant=True,
            address=address,
            strata=Strata.predict,
        )
        out[address], _, _ = TensorField.new(sources, schema=schema, context=Context())

    inputs = TensorDict(source=cast(Any, out), batch_size=[batch_size])
    inputs[root] = TensorDict(source=cast(Any, axes), batch_size=[batch_size])
    return inputs
