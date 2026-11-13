"""Bounded, diagnostic input distributions independent of learned model state."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from relflow.logging import logger
from relflow.structs.drift import Drift
from relflow.structs.enums import Strata

if TYPE_CHECKING:
    from relflow.structs.tree import Address
    from relflow.tensorfields.shared.digest import Digest

__all__ = ["Drift", "Distribution", "Distributions"]


@dataclass
class Distribution:
    """One mergeable feature: a numeric digest or a fixed categorical histogram."""

    digest: Digest | None = None
    counts: np.ndarray | None = None
    encoding: str | None = None

    def __post_init__(self) -> None:
        if (self.digest is None) == (self.counts is None):
            raise ValueError("distribution requires exactly one numeric digest or categorical histogram")
        if self.counts is not None and (
            self.counts.ndim != 1 or self.counts.dtype != np.int64 or len(self.counts) == 0 or np.any(self.counts < 0)
        ):
            raise ValueError("distribution counts must be a nonempty one-dimensional nonnegative int64 histogram")

    @classmethod
    def numeric(cls, values: pa.Array | pa.ChunkedArray) -> Distribution:
        from relflow.tensorfields.shared.digest import Digest

        digest = Digest()
        digest.update(values)
        return cls(digest=digest)

    @classmethod
    def categorical(cls, indices: np.ndarray, *, bins: int, encoding: str | None = None) -> Distribution:
        indices = np.asarray(indices)
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError("categorical distribution bins must be a positive integer")
        if (
            indices.ndim != 1
            or (indices.size > 0 and indices.dtype.kind not in "biu")
            or np.any(indices < 0)
            or np.any(indices >= bins)
        ):
            raise ValueError(f"categorical distribution indices must be a one-dimensional integer array in [0, {bins})")
        return cls(counts=np.bincount(indices.astype(np.int64), minlength=bins).astype(np.int64), encoding=encoding)

    @property
    def count(self) -> int:
        return (
            self.digest.count if self.digest is not None else int(self.counts.sum()) if self.counts is not None else 0
        )

    def merge(self, other: Distribution) -> None:
        if self.encoding != other.encoding:
            raise ValueError("distribution encoding changed; clear its profiles before merging")
        if self.digest is not None and other.digest is not None:
            self.digest.merge(other.digest)
        elif self.counts is not None and other.counts is not None and self.counts.shape == other.counts.shape:
            self.counts += other.counts
        else:
            raise ValueError(
                "distribution feature changed representation; clear its profiles before changing the schema"
            )

    def compare(self, other: Distribution) -> tuple[float, float]:
        """Effect size and a conservative finite-sample allowance.

        Numeric distance approximates the maximum CDF difference, including
        both sides of point masses. Categorical distance is total variation.
        Allowances reduce noisy warnings; digest approximation and repeated
        comparisons mean they carry no statistical significance guarantee.
        """
        if self.encoding != other.encoding:
            raise ValueError("distribution comparison requires matching encodings")
        if not self.count or not other.count:
            raise ValueError("distribution comparison requires observations in both profiles")
        uncertainty = 1 / self.count + 1 / other.count
        if self.digest is not None and other.digest is not None:
            knots = np.unique(
                np.concatenate((self.digest.centroids["mean"].to_numpy(), other.digest.centroids["mean"].to_numpy()))
            )
            with np.errstate(over="ignore"):
                knots = np.unique(np.concatenate((knots, np.nextafter(knots, -np.inf), np.nextafter(knots, np.inf))))
            knots = pa.array(knots[np.isfinite(knots)])
            score = float(np.max(np.abs(self.digest.cdf(knots).to_numpy() - other.digest.cdf(knots).to_numpy())))
            return score, math.sqrt(math.log(2000) * uncertainty / 2)
        if self.counts is not None and other.counts is not None and self.counts.shape == other.counts.shape:
            score = float(np.abs(self.counts / self.count - other.counts / other.count).sum() / 2)
            occupied = int(np.count_nonzero(self.counts + other.counts))
            return score, math.sqrt(max(1, occupied - 1) * uncertainty) / 2 + math.sqrt(
                math.log(2000) * uncertainty / 2
            )
        raise ValueError("cannot compare different distribution representations; clear the field profiles")

    def summary(self) -> dict[str, object]:
        if self.digest is not None:
            return {
                "count": self.count,
                "quantiles": self.digest.quantile(pa.array([0.0, 0.25, 0.5, 0.75, 1.0])).to_pylist()
                if self.count
                else [],
            }
        return {
            "count": self.count,
            "counts": self.counts.tolist() if self.counts is not None else [],
            "encoding": self.encoding,
        }


@dataclass
class Distributions:
    """Rank-local cumulative profiles, keyed by field, stratum, and feature.

    Only training updates the reference. Evaluation inputs may update their
    own profile but cannot update the reference or the model's learned state.
    """

    profiles: dict[str, dict[str, dict[str, Distribution]]] = field(default_factory=dict)
    warned: set[tuple[str, str, str]] = field(default_factory=set)
    signatures: dict[str, object] = field(default_factory=dict)

    def clear(self, *, address: str | None = None, strata: Strata | str | None = None) -> None:
        """Begin a new observation window, or discard an obsolete reference."""
        normalized = None if strata is None else Strata.normalize(strata).value
        for path in list(self.profiles):
            if address is not None and path != address:
                continue
            if normalized is None:
                del self.profiles[path]
            else:
                self.profiles[path].pop(normalized, None)
        self.warned = {
            key
            for key in self.warned
            if not (
                (address is None or key[0] == address)
                and (normalized is None or normalized == "train" or key[1] == normalized)
            )
        }

    def update(
        self,
        incoming: Mapping[Address, Mapping[str, Distribution]],
        *,
        strata: Strata,
        policies: Mapping[Address, Drift | None],
        rank: int = 0,
    ) -> None:
        for address, features in incoming.items():
            policy = policies.get(address)
            if policy is None:
                continue
            by_strata = self.profiles.setdefault(str(address), {})
            current = by_strata.setdefault(strata.value, {})
            for name, distribution in features.items():
                if any(
                    name in existing and existing[name].encoding != distribution.encoding
                    for existing in by_strata.values()
                ):
                    for existing in by_strata.values():
                        existing.pop(name, None)
                    self.warned = {key for key in self.warned if key[0] != address or key[2] != name}
                if name not in current:
                    # Batch payloads can be replayed or retained by the caller.
                    current[name] = copy.deepcopy(distribution)
                else:
                    current[name].merge(distribution)
            reference = by_strata.get("train", {})
            for split, observed in by_strata.items():
                if split == "train":
                    continue
                for name in reference.keys() & observed.keys():
                    baseline, candidate = reference[name], observed[name]
                    key = (str(address), split, name)
                    if key in self.warned or min(baseline.count, candidate.count) < policy.min_samples:
                        continue
                    score, allowance = baseline.compare(candidate)
                    if score <= policy.threshold + allowance:
                        continue
                    self.warned.add(key)
                    logger.bind(
                        component="drift",
                        address=str(address),
                        strata=split,
                        feature=name,
                        reference="train",
                        reference_count=baseline.count,
                        count=candidate.count,
                        distance=score,
                        threshold=policy.threshold,
                        allowance=allowance,
                        scope="rank-local",
                        rank=rank,
                    ).warning(
                        "input distribution differs from training; check the split or adjust this field's drift policy"
                    )

    def state_dict(self) -> dict[str, object]:
        """Keep checkpoint metadata compatible with Torch weights-only loading."""
        return {
            "profiles": {
                address: {
                    strata: {
                        name: {
                            "digest": feature.digest.serialize() if feature.digest is not None else None,
                            "counts": feature.counts.tolist() if feature.counts is not None else None,
                            "encoding": feature.encoding,
                        }
                        for name, feature in features.items()
                    }
                    for strata, features in splits.items()
                }
                for address, splits in self.profiles.items()
            },
            "warned": sorted(self.warned),
            "signatures": copy.deepcopy(self.signatures),
        }

    @classmethod
    def from_state(cls, state: dict) -> Distributions:
        from relflow.tensorfields.shared.digest import Digest

        result = cls()
        for address, splits in state.get("profiles", {}).items():
            result.profiles[address] = {}
            for strata, features in splits.items():
                result.profiles[address][strata] = {
                    name: Distribution(
                        digest=Digest.deserialize(value["digest"]) if value["digest"] is not None else None,
                        counts=np.asarray(value["counts"], dtype=np.int64) if value["counts"] is not None else None,
                        encoding=value.get("encoding"),
                    )
                    for name, value in features.items()
                }
        result.warned = {tuple(key) for key in state.get("warned", [])}
        result.signatures = copy.deepcopy(state.get("signatures", {}))
        return result

    def summary(self) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
        """Return JSON-compatible metadata without raw observations or labels."""
        return {
            address: {
                strata: {name: feature.summary() for name, feature in features.items()}
                for strata, features in splits.items()
            }
            for address, splits in self.profiles.items()
        }
