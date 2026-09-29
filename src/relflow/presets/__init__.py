"""Immutable model presets and their typed classmethod factory."""

from relflow.presets.base import BranchDefaults, LeafDefaults, Preset
from relflow.presets.builtins import LG, MD, SM, XL, XS
from relflow.presets.construction import factory

__all__ = [
    "BranchDefaults",
    "LeafDefaults",
    "Preset",
    "XS",
    "SM",
    "MD",
    "LG",
    "XL",
    "factory",
]
