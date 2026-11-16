"""Frozen architecture configuration owned by Evidence, independent of SDK presets.

CONTROL preserves the historical XS architecture. Its internal name identifies
the policy owner and revision; generators, model fields, training options and
checks remain in each experiment. Import this module only during execution.
"""

import relflow as rf

__all__ = ["CONTROL"]


CONTROL = rf.presets.register(
    rf.presets.Preset(
        name="evidence-control-v1",
        revision=1,
        d_model=64,
        root=rf.presets.BranchDefaults(
            n_layers=1,
            n_heads=2,
            ffnx=8,
            attention="mha",
            dropout=0.0,
            reduction=rf.Attention(n_outputs=1, n_heads=None, n_layers=1, dropout=None, position=True),
        ),
        branch=rf.presets.BranchDefaults(
            n_layers=1,
            n_heads=2,
            ffnx=8,
            attention="mha",
            dropout=0.0,
            reduction=rf.Attention(n_outputs=1, n_heads=None, n_layers=1, dropout=None, position=True),
        ),
        leaf=rf.presets.LeafDefaults(
            n_heads=2,
            ffnx=8,
            pooling="query",
            n_linear=1,
            dropout=0.0,
            decoder_position=None,
        ),
        fit=None,
    )
)
