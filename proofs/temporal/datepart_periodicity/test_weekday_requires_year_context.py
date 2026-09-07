"""Show that day-of-year cannot identify weekday after the year is discarded.

Example
-------
The same ordinal day lands on different weekdays in different years:

```yaml
input:
  day_only:
    - {observed_at: 2001-01-01, visible_day_of_year: 1}
    - {observed_at: 2002-01-01, visible_day_of_year: 1}
  identified:
    - {visible_day_of_year: 1, visible_day_of_week: Monday}
    - {visible_day_of_year: 1, visible_day_of_week: Tuesday}
expected_output:
  target: [Monday, Tuesday]
  day_only_result: ambiguous
  identified_result: at_least_0.95_accuracy
```

The first representation is ambiguous by construction. Adding the visible
weekday coordinate makes the target identifiable.
"""

import pytest

from proofs.temporal.datepart_periodicity.support import accuracy, balanced_weekdays, fit, model

pytestmark = pytest.mark.proof


def test_weekday_requires_a_visible_weekday_or_year_coordinate() -> None:
    """Balanced coordinate collisions vanish when day-of-week is made visible."""

    seed = 47
    train = balanced_weekdays(years=tuple(range(1901, 1936)))
    validate = balanced_weekdays(years=tuple(range(1951, 1986)))
    test = balanced_weekdays(years=tuple(range(2001, 2036)))

    day_only = model(dateparts=("day_of_year",), classes=7, seed=seed)
    fit(day_only, train=train, validate=validate, epochs=14, seed=seed)
    ambiguous_accuracy = accuracy(day_only, test)

    identified = model(dateparts=("day_of_year", "day_of_week"), classes=7, seed=seed)
    fit(identified, train=train, validate=validate, epochs=14, seed=seed)
    identified_accuracy = accuracy(identified, test)

    details = f"day-only={ambiguous_accuracy:.4f}, with-weekday={identified_accuracy:.4f}"
    assert ambiguous_accuracy <= 0.20, details
    assert identified_accuracy >= 0.95, details
    assert identified_accuracy - ambiguous_accuracy >= 0.70, details
