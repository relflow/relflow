"""Learn month phase from day-of-year on dates and years withheld from training.

Example
-------
Only the cyclical day-of-year representation is visible to the model:

```yaml
input:
  rows:
    - {observed_at: 2025-02-04, visible_day_of_year: 35}
    - {observed_at: 2025-11-18, visible_day_of_year: 322}
expected_output:
  target: [February, November]
control:
  permuted_observed_at: accuracy below 0.20
```

Training contains odd dates from earlier years; these even dates and years are
held out. Permuting source dates while keeping the targets must break the
mapping.
"""

import pytest

from proofs.temporal.datepart_periodicity.support import (
    accuracy,
    dates,
    fit,
    model,
    permute_observed_at,
)

pytestmark = pytest.mark.proof


def test_day_of_year_generalizes_month_phase_to_unseen_dates() -> None:
    """Odd training dates interpolate month labels onto even dates in new years."""

    seed = 41
    train = dates((2017, 2018, 2019), parity=1)
    validate = dates((2021,), parity=1)
    test = dates((2023, 2025), parity=0)
    configured = model(dateparts=("day_of_year",), classes=12, seed=seed)
    fit(configured, train=train, validate=validate, epochs=70, seed=seed)

    periodic_accuracy = accuracy(configured, test)
    permuted_accuracy = accuracy(configured, permute_observed_at(test, seed=seed + 1))
    details = f"periodic={periodic_accuracy:.4f}, permuted={permuted_accuracy:.4f}"
    assert periodic_accuracy >= 0.90, details
    assert permuted_accuracy <= 0.20, details
    assert periodic_accuracy - permuted_accuracy >= 0.65, details
