"""Architecture defaults from the frozen arena HPO recipes.

SMALL follows small-regularized-capacity (trial 33); MEDIUM follows
medium-deep-context (trial 8), also used for large tasks. These describe data
regimes, not an ascending parameter ladder. See provenance.json for the exact
source configurations and hashes. Training recipes remain application-owned.
"""

from relflow.presets.base import BranchDefaults, FitDefaults, LeafDefaults, Preset
from relflow.structs.reduction import Attention

__all__ = ["SMALL", "MEDIUM"]


SMALL = Preset(
    name="small",
    d_model=128,
    root=BranchDefaults(n_layers=2, n_heads=4, ffnx=2, dropout=0.0, reduction=Attention()),
    branch=BranchDefaults(n_layers=1, n_heads=4, ffnx=8, dropout=0.0, reduction=Attention(n_outputs=4)),
    leaf=LeafDefaults(n_heads=4, ffnx=8, n_linear=1, dropout=0.0),
    fit=FitDefaults(
        batch_size=64,
        learning_rate=0.0001504508367492947,
        weight_decay=0.01,
        max_epochs=8,
        ema_decay=0.995,
        gradient_clip_val=0.25,
    ),
)


MEDIUM = Preset(
    name="medium",
    d_model=64,
    root=BranchDefaults(n_layers=2, n_heads=8, ffnx=4, dropout=0.0, reduction=Attention()),
    branch=BranchDefaults(n_layers=2, n_heads=4, ffnx=8, dropout=0.0, reduction=Attention(n_outputs=1)),
    leaf=LeafDefaults(n_heads=4, ffnx=8, n_linear=1, dropout=0.0),
    fit=FitDefaults(
        batch_size=512,
        learning_rate=0.00021681002964478338,
        weight_decay=0.0,
        max_epochs=8,
        ema_decay=0.995,
        gradient_clip_val=0.25,
    ),
)
