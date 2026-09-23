from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial, partialmethod
from multiprocessing import Manager
from multiprocessing.managers import ListProxy, SyncManager
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from relflow.distributed import (
    all_gather_object,
    all_reduce_max,
    broadcast_object,
    is_distributed,
    is_rank_zero,
    synchronize_epoch_metrics,
)
from relflow.logging import logger
from relflow.structs.enums import Strata, TensorKey
from relflow.structs.tree import Address
from relflow.tensorfields.shared.counter import Counter

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.helpers.resize import Resize


MIN_OBSERVATIONS = 10
INITIAL_CAPACITY = 1


def capacity(initial: int, required: int) -> int:
    """Grow geometrically without declaring an upper vocabulary limit."""
    return initial << (max(required - 1, 0) // initial).bit_length()


@dataclass(frozen=True)
class VocabularyBatch:
    """An immutable Arrow dictionary for the IDs in one prefetched batch."""

    labels: pa.Array
    size: int

    def resolve(
        self, vocabulary: OnlineVocabularyModel, *, learn: bool, device: torch.device, resize: Resize
    ) -> tuple[torch.Tensor, int]:
        """Admit the current ranks' labels and map local IDs, including unknown."""
        indices = vocabulary.lookup(self.labels)
        size = vocabulary.size
        if learn:
            unseen = indices < 0
            needed = bool(unseen.any())
            if is_distributed():
                needed = bool(all_reduce_max(torch.tensor(needed, dtype=torch.int64, device=device)).item())
            if needed:
                gathered = all_gather_object(pc.filter(self.labels, pa.array(unseen)))
                existing = vocabulary.index()
                additions: dict[Any, int] = {}
                for labels in gathered:
                    for label in labels.to_pylist():
                        if label not in existing and label not in additions:
                            additions[label] = len(existing) + len(additions)
                size = capacity(size, len(existing) + len(additions))
                for position, label in enumerate(self.labels.to_pylist()):
                    if indices[position] < 0:
                        indices[position] = additions[label]
                resize.publish(partial(vocabulary.extend, list(additions)))
        mapping = torch.full((self.size + 1,), -1, dtype=torch.int64)
        mapping[: len(indices)] = torch.from_numpy(indices.copy())
        return mapping, size

    def counts(self, values: torch.Tensor, mapping: torch.Tensor, size: int) -> torch.Tensor:
        """Move pristine exposure to global IDs without counting unused rows."""
        indices = mapping[: values.numel()].to(values.device)
        valid = indices.ge(0) & indices.lt(size)
        result = values.new_zeros(size)
        result.index_add_(0, indices[valid], values[valid])
        return result


class LocalLock:
    """Pickle-friendly local lock used outside multiprocessing data workers."""

    def __init__(self) -> None:
        self._lock = RLock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __getstate__(self) -> dict[str, Any]:
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._lock = RLock()


@dataclass
class VocabularyStorage:
    master: list[Any] | ListProxy[Any]
    lock: Any


class VocabularyState:
    def __init__(
        self,
        storage: VocabularyStorage,
        size: int,
        share: Callable[[], VocabularyStorage] | None = None,
    ):
        self.storage = storage
        self.size: int = size
        self._share = share
        self.vocab: list[Any] = []
        self.index: dict[Any, int] = {}
        self.refresh(force=True)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_share"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def batch(self, values: pa.Array | pa.ChunkedArray) -> tuple[object, object]:
        """Keep worker IDs independent of rank, prefetch order, and live capacity."""
        array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
        while (
            pa.types.is_list(array.type)
            or pa.types.is_large_list(array.type)
            or pa.types.is_fixed_size_list(array.type)
        ):
            array = pc.list_flatten(array)
        labels = pc.unique(pc.drop_null(array))
        size = max(1, len(labels))
        local = OnlineVocabularyModel(size=size)
        local.load_snapshot(labels.to_pylist())
        return local.state, VocabularyBatch(labels=labels, size=size)

    def share(self) -> None:
        if self._share is None:
            return

        self.storage = self._share()
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        master_size = len(self.master)
        vocab_size = len(self.vocab)

        if not force and master_size == vocab_size:
            return

        if force or master_size < vocab_size:
            self.vocab = list(self.master)
            self.index = {word: index for index, word in enumerate(self.vocab)}
            return

        for word in self.master[vocab_size:]:
            if word in self.index:
                continue

            self.index[word] = len(self.vocab)
            self.vocab.append(word)

    @property
    def master(self) -> list[Any] | ListProxy[Any]:
        return self.storage.master

    @property
    def lock(self) -> Any:
        return self.storage.lock

    @property
    def unavailable_index(self) -> int:
        return -1

    def reserve(self, values: Any, *, learn: bool) -> None:
        """Reserve every scalar token found in a JSON-like nested value."""
        # A checkpoint restore can shrink the master while an existing loader
        # still holds this state. Refresh before consulting its cached index.
        self.refresh()
        if not learn:
            return

        candidates: list[Any] = []
        seen: set[Any] = set()
        for word in self.tokens(values):
            if word is None or word in self.index or word in seen:
                continue

            seen.add(word)
            candidates.append(word)

        if not candidates:
            return

        with self.lock:
            self.refresh()
            for word in candidates:
                if word in self.index:
                    continue

                self.index[word] = len(self.vocab)
                self.vocab.append(word)
                self.master.append(word)
            self.size = capacity(self.size, len(self.vocab))

    def encode(self, word: Any) -> int | None:
        if word is None:
            return None

        if word not in self.index:
            self.refresh()

        return self.index.get(word, self.unavailable_index)

    def indices(self, values: pa.Array | pa.ChunkedArray, *, learn: bool) -> np.ndarray:
        """Encode one whole non-null Arrow token column through unique values."""

        array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
        if array.null_count:
            raise ValueError("vocabulary token arrays cannot contain nulls")
        if not len(array):
            self.reserve((), learn=learn)
            return np.empty(0, dtype=np.int64)

        unique = pc.unique(array)
        candidates = unique.to_pylist()
        self.reserve(candidates, learn=learn)
        encoded = pa.array(
            [self.encode(candidate) for candidate in candidates],
            type=pa.int64(),
        )
        positions = pc.index_in(array, value_set=unique)
        if positions.null_count:
            raise RuntimeError("Arrow vocabulary lookup failed to resolve a source token")
        selected = pc.take(encoded, positions)
        return selected.to_numpy(zero_copy_only=False)

    def tokens(self, values: Any) -> Iterable[Any]:
        if values is None:
            return

        if isinstance(values, str | bytes):
            yield values
            return

        if isinstance(values, Iterable):
            for value in values:
                yield from self.tokens(value)
            return

        yield values

    def __len__(self) -> int:
        self.refresh()
        return len(self.vocab)


class OnlineVocabularyModel(torch.nn.Module):
    @classmethod
    def from_model(cls, module: Model) -> dict[Address, OnlineVocabularyModel]:
        resources: dict[Address, OnlineVocabularyModel] = {}

        for address, node in module.nodes.items():
            embedder = getattr(node, "embedder", None)
            vocabulary = getattr(embedder, "vocab", None)
            if isinstance(vocabulary, cls):
                resources[cast(Address, address)] = vocabulary

        return resources

    def __init__(self, size: int = INITIAL_CAPACITY):
        super().__init__()

        self.size: int = size
        self.manager: SyncManager | None = None
        self.master: list[Any] | ListProxy[Any] = []
        self.lock: Any = LocalLock()
        self._snapshot_cache: list[Any] | None = None
        self._snapshot_size: int = -1
        self._labels_cache: pa.Array | None = None
        self._labels_source: list[Any] | None = None
        self._index_cache: dict[Any, int] = {}
        self._index_source: list[Any] | None = None
        self._index_size: int = 0

    def rebuild_state(self, previous: torch.nn.Module) -> dict[str, Any]:
        if not isinstance(previous, OnlineVocabularyModel):
            return {}
        return {"capacity": previous.size, "vocabulary": previous.snapshot()}

    @property
    def storage(self) -> VocabularyStorage:
        return VocabularyStorage(
            master=self.master,
            lock=self.lock,
        )

    @property
    def is_shared(self) -> bool:
        return self.manager is not None

    def share(self) -> None:
        """Move vocabulary state into multiprocessing-safe storage."""
        if self.manager is not None:
            return

        master = list(self.master)
        self.manager = Manager()
        self.master = self.manager.list(master)
        self.lock = self.manager.Lock()
        self._snapshot_cache = None
        self._snapshot_size = -1

    def freeze(self) -> None:
        """Snapshot vocabulary state back into local, read-optimized storage."""
        if self.manager is None:
            return

        manager = self.manager
        self.master = list(self.master)
        self.lock = LocalLock()
        self.manager = None
        self._snapshot_cache = None
        self._snapshot_size = -1
        manager.shutdown()

    def shared_state(self) -> VocabularyStorage:
        self.share()
        return self.storage

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "vocabulary"] = list(self.master)
        destination[prefix + "capacity"] = self.size

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "vocabulary"
        saved_size = state_dict.pop(prefix + "capacity", self.size)
        if not isinstance(saved_size, int) or isinstance(saved_size, bool) or saved_size < 1:
            error_msgs.append(f"{prefix}capacity must be a positive integer, got {saved_size!r}")
        else:
            self.size = saved_size
        if key in state_dict:
            vocab: list[Any] = state_dict.pop(key)
            self.load_snapshot(vocab)
            self._snapshot_cache = None
            self._snapshot_size = -1
        else:
            missing_keys.append(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def state(self) -> VocabularyState:
        return VocabularyState(
            storage=self.storage,
            size=self.size,
            share=self.shared_state,
        )

    def snapshot(self) -> list[Any]:
        size = len(self.master)
        if self._snapshot_cache is None or self._snapshot_size > size:
            self._snapshot_cache = list(self.master)
        elif self._snapshot_size < size:
            self._snapshot_cache.extend(self.master[self._snapshot_size : size])
        self._snapshot_size = size

        return self._snapshot_cache

    def labels(self) -> pa.Array:
        """Return canonical string labels, rebuilding only after vocabulary changes."""
        snapshot = self.snapshot()
        if self._labels_cache is None or self._labels_source is not snapshot:
            self._labels_cache = pa.array([str(value) for value in snapshot], type=pa.large_string())
            self._labels_source = snapshot
        elif len(self._labels_cache) != len(snapshot):
            self._labels_cache = pa.array([str(value) for value in snapshot], type=pa.large_string())

        return self._labels_cache

    def index(self) -> dict[Any, int]:
        """Extend the model's lookup cache only for newly committed labels."""
        snapshot = self.snapshot()
        if self._index_source is not snapshot:
            self._index_cache = {}
            self._index_size = 0
            self._index_source = snapshot
        for index in range(self._index_size, len(snapshot)):
            self._index_cache[snapshot[index]] = index
        self._index_size = len(snapshot)
        return self._index_cache

    def lookup(self, values: pa.Array) -> np.ndarray:
        index = self.index()
        return np.asarray([index.get(value, -1) for value in values.to_pylist()], dtype=np.int64)

    def extend(self, labels: list[Any]) -> None:
        """Publish labels already bound to their model-owned rows."""
        with self.lock:
            index = self.index()
            for word in labels:
                if word in index:
                    continue

                index[word] = len(self.master)
                self.master.append(word)

            self.size = capacity(self.size, len(self.master))

    def load_snapshot(self, vocabulary: list[Any]) -> None:
        with self.lock:
            self.size = capacity(self.size, len(vocabulary))
            self.master[:] = vocabulary

        self._snapshot_cache = None
        self._snapshot_size = -1


class VocabularySyncCallback(Callback):
    """Synchronize vocabularies and warn about low exposure before evaluation.

    Exposure checks use an embedder's shared ``vocab`` and ``counters['content']``
    resources. Extensions without a content Counter retain synchronization only.
    """

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Match initial label metadata to DDP's rank-zero parameter broadcast."""
        if not is_distributed():
            return
        resources = OnlineVocabularyModel.from_model(cast("Model", pl_module))
        snapshots = {address: vocabulary.snapshot() for address, vocabulary in resources.items()}
        for address, snapshot in broadcast_object(snapshots, src=0).items():
            resources[address].load_snapshot(snapshot)

    @torch.no_grad()
    def warn(self, trainer: Trainer, pl_module: LightningModule, strata: Strata) -> None:
        module = cast("Model", pl_module)
        resources: list[tuple[Address, OnlineVocabularyModel, Counter]] = []
        for address, vocabulary in sorted(
            OnlineVocabularyModel.from_model(module).items(), key=lambda item: str(item[0])
        ):
            counters = getattr(module.nodes[address].embedder, "counters", {})
            if TensorKey.content.name not in counters:
                continue
            counter = counters[TensorKey.content.name]
            if isinstance(counter, Counter):
                resources.append((address, vocabulary, counter))

        if resources and is_distributed():
            # Validation can start before epoch-end counter synchronization.
            # All ranks participate, even with empty local vocabularies.
            synchronize_epoch_metrics(trainer)
            for _, _, counter in resources:
                counter.sync()

        if not is_rank_zero():
            return

        for address, vocabulary, counter in resources:
            size = len(vocabulary.master)
            if not size:
                continue
            # Counters start at one for smoothing, not an observed example.
            counts = counter.counts[:size] - 1
            underobserved = int(counts.lt(MIN_OBSERVATIONS).sum().item())
            if not underobserved:
                continue
            logger.bind(
                component="vocabulary",
                address=str(address),
                strata=strata.value,
                underobserved=underobserved,
                size=size,
                threshold=MIN_OBSERVATIONS,
                minimum_observations=int(counts.min().item()),
            ).warning(
                f"vocabulary entries have fewer than {MIN_OBSERVATIONS} training observations; "
                "cold-start representations may affect metrics or predictions. "
                "Continue training and evaluate low-exposure labels separately."
            )

    if TYPE_CHECKING:
        on_validation_start = Callback.on_validation_start
        on_test_start = Callback.on_test_start
        on_predict_start = Callback.on_predict_start
    else:
        on_validation_start = partialmethod(warn, strata=Strata.validate)
        on_test_start = partialmethod(warn, strata=Strata.test)
        on_predict_start = partialmethod(warn, strata=Strata.predict)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module = cast("Model", pl_module)
        for vocabulary in OnlineVocabularyModel.from_model(pl_module).values():
            vocabulary.freeze()


__all__ = [
    "INITIAL_CAPACITY",
    "OnlineVocabularyModel",
    "VocabularyState",
    "VocabularySyncCallback",
]
