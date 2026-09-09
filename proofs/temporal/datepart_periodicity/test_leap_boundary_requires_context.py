"""Characterize the February/March ambiguity at ordinal day 60.

Example
-------
Leap day and an ordinary March boundary share the same ordinal coordinate:

```yaml
input:
  day_only:
    - {observed_at: 2024-02-29, visible_day_of_year: 60}
    - {observed_at: 2023-03-01, visible_day_of_year: 60}
  identified:
    - {visible_day_of_year: 60, visible_week_of_month: 5}
    - {visible_day_of_year: 60, visible_week_of_month: 1}
expected_output:
  target: [February, March]
  day_only_accuracy: 0.50
  identified_accuracy: at_least_0.95
```

Day-of-year alone must remain at chance; the second coordinate resolves the
collision.
"""

import pytest

from proofs.temporal.datepart_periodicity.support import accuracy, fit, leap_boundary, model

pytestmark = pytest.mark.proof


def test_week_of_month_resolves_the_leap_year_month_boundary() -> None:
    """Day 60 needs another coordinate to distinguish February 29 from March 1."""

    seed = 53
    train = leap_boundary(years=tuple(range(1901, 1951)), rows=512)
    validate = leap_boundary(years=tuple(range(1951, 2001)), rows=128)
    test = leap_boundary(years=tuple(range(2001, 2051)), rows=256)

    day_only = model(dateparts=("day_of_year",), classes=2, seed=seed)
    fit(day_only, train=train, validate=validate, epochs=10, seed=seed)
    ambiguous_accuracy = accuracy(day_only, test)

    identified = model(dateparts=("day_of_year", "week_of_month"), classes=2, seed=seed)
    fit(identified, train=train, validate=validate, epochs=20, seed=seed)
    identified_accuracy = accuracy(identified, test)

    details = f"day-only={ambiguous_accuracy:.4f}, with-week-of-month={identified_accuracy:.4f}"
    assert 0.49 <= ambiguous_accuracy <= 0.51, details
    assert identified_accuracy >= 0.95, details
    assert identified_accuracy - ambiguous_accuracy >= 0.40, details
