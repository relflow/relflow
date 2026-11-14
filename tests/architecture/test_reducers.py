import typing

import torch

import relflow as rf
from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.architecture.reducers import cross, mean, reducers
from relflow.structs.reduction import ReductionConfig


def test_every_reduction_setting_has_a_reducer():
    union = typing.get_args(typing.get_args(ReductionConfig)[0])
    assert set(union) == set(reducers)


def test_reducers_build_todays_pools():
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        events=rf.Branch(length=3, n_heads=2, amount=rf.Number),
    )
    branch = schema.branches[rf.Address("events")]
    pool = mean(rf.Mean(), branch=branch, schema=schema, capacity=3, dropout=0.1)
    assert isinstance(pool, MeanPool) and pool.n_context == 1 and pool.mass_capacity == 3
    pool = cross(rf.Attention(n_outputs=2), branch=branch, schema=schema, capacity=3, dropout=0.1)
    assert isinstance(pool, LearnedQueryCrossAttention) and pool.n_context == 2
    assert pool.blocks[0].attention.nhead == 2 and pool.mass_capacity == 3


def test_mean_pool_ignores_the_shared_reducer_keywords():
    pool = MeanPool(n_context=1, d_model=4)
    memory = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(0))
    present = torch.tensor([[True, False, True], [True, True, True]])
    plain = pool(memory, present=present)
    shared = pool(memory, present=present, context=torch.ones(2, 1, 4), evidence=memory, memory_positions=None)
    torch.testing.assert_close(shared, plain, rtol=0, atol=0)
