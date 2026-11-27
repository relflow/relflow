"""Keyword declarations for running the same Evidence protocol with either SDK."""

from types import MappingProxyType

import relflow as rf

__all__ = ["TARGET", "reconstruct"]

# Source checkouts can share a package version while exposing different schemas.
TARGET = MappingProxyType({"predict": True} if "predict" in rf.Leaf.model_fields else {"mask": True})


def reconstruct(selector: float | str, *also: float | str) -> dict:
    """Use the union of the given sampling rates and Boolean query paths as answers."""
    rules = [{"query": value} if isinstance(value, str) else {"rate": value} for value in (selector, *also)]
    if "predict" in TARGET:
        return {"task": tuple(rf.reconstruct(**rule) for rule in rules)}
    return {"mask": tuple(rf.Mask(**rule, reconstruct=True) for rule in rules)}
