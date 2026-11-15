from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

from relflow.data.rows import advance, last, runs, trailing


def test_last_selects_the_final_rows_of_each_parent():
    placement = np.array([0, 1, 2, 3, 5, 6])  # extent 4: parent 0 has 4 rows, parent 1 has 2
    assert last(placement, extent=4, count=2).tolist() == [False, False, True, True, True, True]
    assert last(placement, extent=4, count=5).tolist() == [True] * 6


def test_last_of_no_records_is_empty():
    assert last(np.empty(0, dtype=np.int64), extent=0, count=2).tolist() == []


def test_runs_never_cross_a_parent_boundary():
    placement = np.array([0, 1, 2, 3])  # extent 2: two parents of 2 rows
    starts = np.array([False, True, False, False])
    assert runs(placement, extent=2, starts=starts, length=5).tolist() == [False, True, False, False]


def test_runs_cover_length_rows_from_each_start():
    placement = np.arange(8)
    starts = np.array([False, True] + [False] * 6)
    assert runs(placement, extent=8, starts=starts, length=3).tolist() == [False, True, True, True] + [False] * 4


def test_trailing_marks_rows_after_the_last_observed_row():
    placement = np.array([0, 1, 2, 4, 5])  # extent 4: parent 0 has 3 rows, parent 1 has 2
    observed = np.array([True, False, False, False, False])
    assert trailing(placement, extent=4, observed=observed).tolist() == [False, True, True, False, False]


def test_advance_adds_steps_in_the_column_type():
    assert advance(pa.array([7.0, 7.0]), steps=np.array([1, 2]), step=0.5).to_pylist() == [7.5, 8.0]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    stamps = pa.array([start, start], type=pa.timestamp("us", tz="UTC"))
    moved = advance(stamps, steps=np.array([1, 2]), step=timedelta(hours=1))
    # Compare epoch microseconds: zone-aware to_pylist needs tzdata, which Windows lacks.
    hour = 3_600_000_000
    assert moved.type == stamps.type
    assert moved.cast(pa.int64()).to_pylist() == (stamps.cast(pa.int64()).to_numpy() + [hour, 2 * hour]).tolist()
    with pytest.raises(pa.ArrowInvalid):
        advance(pa.array([1, 2], type=pa.int64()), steps=np.array([1, 1]), step=0.5)


def test_advance_moves_dates_by_whole_days_only():
    days = pa.array([date(2026, 1, 1)], type=pa.date32())
    assert advance(days, steps=np.array([2]), step=timedelta(days=1)).to_pylist() == [date(2026, 1, 3)]
    with pytest.raises(pa.ArrowInvalid):
        advance(days, steps=np.array([1]), step=timedelta(hours=1))


def test_advance_rejects_a_step_that_does_not_suit_the_column():
    with pytest.raises(ValueError, match="time_step"):
        advance(pa.array([1.0]), steps=np.array([1]), step=timedelta(hours=1))
    with pytest.raises(ValueError, match="time_step"):
        advance(pa.array([datetime(2026, 1, 1)]), steps=np.array([1]), step=0.5)
    with pytest.raises(TypeError, match="string"):
        advance(pa.array(["a"]), steps=np.array([1]), step=1.0)
