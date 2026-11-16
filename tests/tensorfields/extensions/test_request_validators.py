import pytest

import relflow as rf


def test_category_topk_rejects_non_positive():
    with pytest.raises(ValueError, match="topk values must be positive"):
        rf.Category(topk=[0])


def test_category_topk_is_independent_of_vocabulary_storage():
    assert rf.Category(topk=[10000, 8, 10000]).topk == [8, 10000]


def test_category_rejects_number_only_n_bands_option():
    with pytest.raises(ValueError, match="n_bands"):
        rf.Category(n_bands=8)


def test_set_threshold_rejects_values_above_one():
    with pytest.raises(ValueError, match="less than or equal to 1"):
        rf.Set(threshold=1.1)


def test_boolean_threshold_rejects_values_above_one():
    with pytest.raises(ValueError, match="less than or equal to 1"):
        rf.Boolean(threshold=1.1)


@pytest.mark.parametrize(
    ("threshold", "match"),
    [
        ([], "at least 1 item"),
        ([-0.1, 0.5], "greater than or equal to 0"),
        ([0.5, 1.1], "less than or equal to 1"),
    ],
)
def test_boolean_threshold_rejects_invalid_lists(threshold: list[float], match: str):

    with pytest.raises(ValueError, match=match):
        rf.Boolean(threshold=threshold)


def test_dateparts_dateparts_reject_duplicates():
    with pytest.raises(ValueError, match="dateparts must be unique"):
        rf.DateParts(dateparts=["day_of_week", "day_of_week"])


def test_dateparts_normalizes_friendly_datepart_names():
    request = rf.DateParts(dateparts=["Day Of Week", "month-of-year", "HourOfDay"])

    assert [datepart.value for datepart in request.dateparts] == [
        "day_of_week",
        "month_of_year",
        "hour_of_day",
    ]


def test_dateparts_unknown_name_suggests_canonical_value():
    with pytest.raises(ValueError, match="did you mean 'day_of_week'"):
        rf.DateParts(dateparts=["day of wek"])


def test_dateparts_pattern_rejects_invalid_tokens():
    with pytest.raises(ValueError, match="is not a valid format pattern"):
        rf.DateParts(dateparts=["day_of_week"], pattern="%Q-%m-%d")


def test_dateparts_pattern_accepts_valid_format():
    request = rf.DateParts(dateparts=["day_of_week", "month_of_year"], pattern="%Y-%m-%d")
    assert request.pattern == "%Y-%m-%d"
