from __future__ import annotations

import enum
from collections.abc import Mapping
from typing import Literal, TypeAlias, TypeVar

T = TypeVar("T")
DefaultT = TypeVar("DefaultT")


class Tokens(enum.IntEnum):
    valued = 0
    null = 1
    padded = 2
    masked = 3
    other = 4


class Strata(enum.StrEnum):
    train = "train"
    validate = "validate"
    test = "test"
    predict = "predict"

    @classmethod
    def normalize(cls, value: "Strata | str") -> "Strata":
        if isinstance(value, cls):
            return value

        # Preserve coercion for callers outside the checked Python API.
        return cls(str(value).strip().lower())  # pyrefly: ignore [unnecessary-type-conversion]

    @classmethod
    def expand(cls, value: T | Mapping[Strata | str, T], *, default: DefaultT) -> dict[Strata, T | DefaultT]:
        if isinstance(value, Mapping):
            normalized: dict[Strata, T | DefaultT] = {strata: default for strata in cls}
            for key, item in value.items():
                normalized[cls.normalize(key)] = item
            return normalized

        return {strata: value for strata in cls}


class TensorKey(enum.StrEnum):
    value = "value"
    content = "content"
    cluster = "cluster"
    state = "state"
    present = "present"
    trainable = "trainable"
    targets = "targets"
    intervals = "intervals"
    probability = "probability"
    topk = "topk"
    embedding = "embedding"
    inferred = "inferred"


class Metric(enum.StrEnum):
    accuracy = "accuracy"
    auc = "auc"
    precision = "precision"
    recall = "recall"
    specificity = "specificity"
    loss = "loss"
    sigma = "sigma"
    throughput = "throughput"
    mae = "mae"
    rmse = "rmse"


class AttentionMode(enum.StrEnum):
    mha = "mha"
    gqa = "gqa"
    mqa = "mqa"

    def kv_heads(self, n_heads: int) -> int:
        match self:
            case AttentionMode.mha:
                return n_heads
            case AttentionMode.gqa:
                return max(1, n_heads // 2)
            case AttentionMode.mqa:
                return 1


class Overflow(enum.StrEnum):
    head = "head"
    tail = "tail"
    error = "error"


AttentionInput: TypeAlias = AttentionMode | Literal["mha", "gqa", "mqa"] | None
OverflowInput: TypeAlias = Overflow | Literal["head", "tail", "error"]
StrataInput: TypeAlias = Strata | Literal["train", "validate", "test", "predict"]


class Component(enum.StrEnum):
    Request = "Request"
    Embedder = "Embedder"
    Decoder = "Decoder"
    TensorField = "TensorField"
    observe = "observe"
    learn = "learn"
    bind = "bind"
    loss = "loss"
    output = "output"
    write = "write"
