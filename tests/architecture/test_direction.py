import pytest
import torch

from relflow.architecture.direction import rules, stepped
from relflow.structs.enums import Direction


def test_every_direction_has_a_rule():
    assert set(rules) == set(Direction)


def test_bidirectional_rule_keeps_padding_only():
    assert rules[Direction.bidirectional](steps=torch.arange(4).expand(2, 4)) is None


def test_causal_rule_lets_tokens_of_one_step_see_each_other():
    allowed = rules[Direction.causal](steps=torch.tensor([[0, 0, 1, 1, 2]]))
    assert allowed[0, 1].tolist() == [True, True, False, False, False]
    assert allowed[0, 2].tolist() == [True, True, True, True, False]


def test_causal_rule_requires_steps():
    with pytest.raises(ValueError, match="causal.*step ids"):
        rules[Direction.causal](steps=None)


def test_every_direction_has_a_rule_and_a_stepped_entry():
    assert set(rules) == set(Direction) == set(stepped)
