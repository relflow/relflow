"""Compose weekday and hour phases instead of relying on either marginal.

Example
-------
Neither weekday nor hour alone separates all three representative rows:

```yaml
input:
  windows:
    - {visible_day_of_week: Monday, visible_hour_of_day: 10}
    - {visible_day_of_week: Monday, visible_hour_of_day: 20}
    - {visible_day_of_week: Saturday, visible_hour_of_day: 10}
expected_output:
  target: [open, closed, closed]
  day_only: insufficient
  hour_only: insufficient
  weekday_and_hour: at_least_0.95_accuracy
```

Only the composed weekday-and-hour representation identifies the business
window.
"""

from datetime import date

import pytest

from proofs.temporal.datepart_periodicity.support import accuracy, business_windows, fit, model

pytestmark = pytest.mark.proof


def test_business_window_requires_weekday_and_hour_together() -> None:
    """Two cyclic coordinates resolve matched workday and work-hour near misses."""

    seed = 59
    train = business_windows(start=date(2017, 1, 2), weeks=52, rows=1024, seed=seed + 1)
    validate = business_windows(start=date(2021, 1, 4), weeks=26, rows=512, seed=seed + 2)
    test = business_windows(start=date(2025, 1, 6), weeks=52, rows=1024, seed=seed + 3)

    day_only = model(dateparts=("day_of_week",), classes=2, seed=seed)
    fit(day_only, train=train, validate=validate, epochs=20, seed=seed)
    day_accuracy = accuracy(day_only, test)

    hour_only = model(dateparts=("hour_of_day",), classes=2, seed=seed)
    fit(hour_only, train=train, validate=validate, epochs=20, seed=seed)
    hour_accuracy = accuracy(hour_only, test)

    composed = model(dateparts=("day_of_week", "hour_of_day"), classes=2, seed=seed)
    fit(composed, train=train, validate=validate, epochs=20, seed=seed)
    composed_accuracy = accuracy(composed, test)

    details = f"day-only={day_accuracy:.4f}, hour-only={hour_accuracy:.4f}, composed={composed_accuracy:.4f}"
    assert day_accuracy <= 0.82, details
    assert hour_accuracy <= 0.82, details
    assert composed_accuracy >= 0.95, details
    assert composed_accuracy - max(day_accuracy, hour_accuracy) >= 0.15, details
