"""Calibrate the proof harness against paired signal and no-signal models.

Claim
-----
Held-out metrics distinguish a learnable relationship from independent
synthetic noise.

Use
---
Generate train, validation, and test rows from independent random streams, and
pair every positive proof with a no-signal model that has the same schema and
training budget. Run this harness check before interpreting a harder proof
failure.

Avoid
-----
Do not judge a proof from training metrics, reuse latent units across splits,
or omit the no-signal control.

Why
---
A complex proof is meaningful only after the harness demonstrates both that it
can learn an obvious signal and that noise remains at chance.

Example
-------
The mixed fields are distractions; only the relationship of ``leak`` to the
target changes:

```yaml
input:
  signal_rows:
    - {x: 0.2, segment: segment-1, tags: [red], leak: false}
    - {x: -0.7, segment: segment-3, tags: [round, hot], leak: true}
expected_output:
  signal_rows:
    - {target: false}
    - {target: true}
  held_out_auc: near_1.0
control:
  leak: independent_random_boolean
  schema_and_training_budget: unchanged
expected_control_output:
  held_out_auc: near_0.5
```

Protocol and gate
-----------------
Train identical mixed-field models on 1,024 rows, validate on 512, and evaluate
on 2,048 held-out rows. Only whether ``leak`` equals the balanced Boolean target
differs. The held-out signal AUC must be at least 0.98, noise AUC must remain in
[0.42, 0.58], and their gap must be at least 0.40.

Status and current evidence
---------------------------
This proof is provisional. For one seed, the signal model clears ``AUC >= 0.98``
and the otherwise identical independent-noise model remains in the chance
band. Train, validation, and test rows come from independent random streams.

Further work
------------
Reconcile the current 2,048 held-out rows with the 4,096 rows specified for
this proof, and record the signal and noise distributions over repeated data
and model seeds.

Promotion criteria
------------------
Pass three paired seeds in the core proof, calibrate the frozen threshold with
at least ten seeds, and retain a material signal-versus-noise gap for every
core seed.

Run with::

    uv run pytest -n 0 proofs/calibration/signal_detection/test_signal_vs_noise.py -q
"""

from __future__ import annotations

import pytest

from proofs.calibration.signal_detection.support import fit_and_test

pytestmark = pytest.mark.proof


def test_held_out_metrics_distinguish_signal_from_noise() -> None:
    signal_auc = fit_and_test(signal=True)
    noise_auc = fit_and_test(signal=False)

    assert signal_auc >= 0.98, f"positive-control AUC did not reach 0.98: {signal_auc:.4f}"
    assert 0.42 <= noise_auc <= 0.58, f"no-signal AUC escaped the calibrated chance band: {noise_auc:.4f}"
    assert signal_auc - noise_auc >= 0.40, (
        f"proof harness did not separate signal and noise: signal={signal_auc:.4f}, noise={noise_auc:.4f}"
    )
