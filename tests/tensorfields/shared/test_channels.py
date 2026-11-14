import pytest

from relflow.tensorfields.shared.channels import width


def test_width_fills_n_dim_from_a_tuple_or_list_query():
    assert width({"query": ("a", "b", "c")})["n_dim"] == 3
    assert width({"query": ["a", "b"]})["n_dim"] == 2


def test_width_keeps_inputs_without_a_tuple_query_and_never_mutates():
    data = {"query": "a", "n_dim": 4}
    assert width(data) == {"query": "a", "n_dim": 4}
    source = {"query": ("a", "b")}
    width(source)
    assert source == {"query": ("a", "b")}


def test_width_rejects_a_different_explicit_n_dim():
    with pytest.raises(ValueError, match=r"n_dim=2 differs from the 3 sources"):
        width({"query": ("a", "b", "c"), "n_dim": 2})
