"""Named preset policies selected at model construction time."""

from relflow.presets.base import Preset
from relflow.presets.builtins import MEDIUM, SMALL

__all__ = ["get", "register"]

PRESETS: dict[str, Preset] = {profile.name: profile for profile in (SMALL, MEDIUM)}


def get(name: str, /) -> Preset:
    """Select a registered policy without capturing an import-time snapshot."""
    if not isinstance(name, str):
        raise TypeError(f"preset name must be a string, got {type(name).__name__}")
    try:
        return PRESETS[name]
    except KeyError as error:
        raise ValueError(f"unknown preset {name!r}; choose a registered name from {sorted(PRESETS)!r}") from error


def register(profile: Preset, /) -> Preset:
    """Validate and copy a custom policy before publishing its unique name."""
    if not isinstance(profile, Preset):
        raise TypeError(f"preset registration requires a Preset, got {type(profile).__name__}")
    profile = Preset.model_validate(profile.model_dump(mode="python", round_trip=True))
    if profile.name in PRESETS:
        raise ValueError(f"preset {profile.name!r} is already registered; choose a distinct name")
    PRESETS[profile.name] = profile
    return profile
