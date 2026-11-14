import numpy as np

from relflow.data.rows import last, runs


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
