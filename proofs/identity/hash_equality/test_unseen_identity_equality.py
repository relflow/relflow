"""Infer equality for Hash identities never observed during training.

Claim
-----
Two Hash fields can express equality for identities absent from the training
vocabulary.

Use
---
Use Hash when observation-local identity equality must generalize to unseen
strings across fields.

Avoid
-----
Do not use independent Category vocabularies for disjoint unseen identity
universes. Both values collapse to unavailable content. Hash is also not a
persistent vocabulary or an invertible identity representation.

Why
---
Hash supplies compatible batch-local representations across fields, whereas
Category is intentionally a persistent learned vocabulary.

Example
-------
The test namespace is absent from training, but equality is still observable:

```yaml
input:
  pairs:
    - {left_id: test-17, right_id: test-17}
    - {left_id: test-17, right_id: test-93}
expected_output:
  equal: [true, false]
controls:
  shuffled_targets: chance-level predictions
  unseen_category_values: no reliable equality signal
```

Protocol and gate
-----------------
Train on 4,096 identity pairs, validate on 1,024, and test on 4,096 from
disjoint namespaces for 20 deterministic CPU epochs. Compare Hash against a
target-shuffle control and a matched Category model. Hash equality AUC must be
at least 0.95, shuffled-control AUC must remain in [0.42, 0.58], Category OOV
AUC must be no more than 0.65, and the Hash/control gap must be at least 0.35.

Status and current evidence
---------------------------
This proof is provisional. The unseen-identity equality route and both
negative controls pass for one seed: two colocated Hash fields clear
``AUC >= 0.95`` on a disjoint identity namespace, shuffled labels remain at
chance, and matched Category fields do not infer equality after both values
become OOV.

Further work
------------
Add a metamorphic control that applies the same unseen-ID bijection to both
fields and preserves predictions. Record Hash, shuffled, and Category-OOV
distributions across seeds.

Promotion criteria
------------------
Pass three paired core seeds and a calibration panel of at least ten while
retaining both the chance control and Category boundary.

Run with::

    uv run pytest -n 0 proofs/identity/hash_equality/test_unseen_identity_equality.py -q
"""

from __future__ import annotations

import pytest

from proofs.identity.hash_equality.support import records, score, train

pytestmark = pytest.mark.proof


def test_hash_equality_generalizes_to_unseen_identities() -> None:
    seed = 29
    train_rows = records(rows=4096, seed=seed + 1, namespace="train")
    validate_rows = records(rows=1024, seed=seed + 2, namespace="validate")
    test = records(rows=4096, seed=seed + 3, namespace="test")
    control = records(rows=4096, seed=seed + 3, namespace="test", shuffle_targets=True)

    hash_model = train(
        identity="hash",
        train_rows=train_rows,
        validate_rows=validate_rows,
        seed=seed,
    )
    category_model = train(
        identity="category",
        train_rows=train_rows,
        validate_rows=validate_rows,
        seed=seed,
    )

    equality_auc = score(hash_model, test)
    control_auc = score(hash_model, control)
    category_oov_auc = score(category_model, test)
    assert equality_auc >= 0.95, f"unseen Hash equality AUC did not reach 0.95: {equality_auc:.4f}"
    assert 0.42 <= control_auc <= 0.58, f"shuffled equality control escaped chance: {control_auc:.4f}"
    assert category_oov_auc <= 0.65, (
        f"Category appeared to infer equality after both disjoint IDs became OOV: AUC={category_oov_auc:.4f}"
    )
    assert equality_auc - control_auc >= 0.35, (
        f"Hash equality did not separate from its control: equality={equality_auc:.4f}, control={control_auc:.4f}"
    )
