import torch

from relflow.architecture.direction import rules
from relflow.structs.enums import Direction


def test_every_direction_has_a_rule():
    assert set(rules) == set(Direction)


def test_bidirectional_rule_keeps_padding_only():
    assert rules[Direction.bidirectional](steps=torch.arange(4).expand(2, 4)) is None
