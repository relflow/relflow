"""Shared tensor types used after Arrow coalescing."""

from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from tensordict import TensorDict

from relflow.structs.enums import StrataInput
from relflow.structs.tree import Address

T = TypeVar("T")

EncodedInput: TypeAlias = TensorDict
InterprocessEncodingContext: TypeAlias = dict[Address, object]
StratumConfig: TypeAlias = T | Mapping[StrataInput | str, T]

__all__ = ["EncodedInput", "InterprocessEncodingContext", "StratumConfig"]
