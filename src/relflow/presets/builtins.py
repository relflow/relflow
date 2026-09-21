"""Initial immutable configurations for the five model sizes."""

from relflow.presets.base import BranchDefaults, LeafDefaults, Preset
from relflow.structs.reduction import Attention

__all__ = ["XS", "SM", "MD", "LG", "XL"]


XS = Preset(
    name="xs",
    d_model=64,
    root=BranchDefaults(n_layers=1, n_heads=2, reduction=Attention(n_outputs=1)),
    branch=BranchDefaults(n_layers=1, n_heads=2, reduction=Attention(n_outputs=1)),
    leaf=LeafDefaults(n_heads=2),
)
SM = Preset(
    name="sm",
    d_model=128,
    root=BranchDefaults(n_layers=2, n_heads=4, reduction=Attention(n_outputs=2)),
    branch=BranchDefaults(n_layers=1, n_heads=4, reduction=Attention(n_outputs=2)),
    leaf=LeafDefaults(n_heads=4),
)
MD = Preset(
    name="md",
    d_model=256,
    root=BranchDefaults(n_layers=3, n_heads=8, reduction=Attention(n_outputs=4)),
    branch=BranchDefaults(n_layers=2, n_heads=8, reduction=Attention(n_outputs=4)),
    leaf=LeafDefaults(n_heads=8),
)
LG = Preset(
    name="lg",
    d_model=384,
    root=BranchDefaults(n_layers=4, n_heads=8, reduction=Attention(n_outputs=8)),
    branch=BranchDefaults(n_layers=2, n_heads=8, reduction=Attention(n_outputs=4)),
    leaf=LeafDefaults(n_heads=8),
)
XL = Preset(
    name="xl",
    d_model=512,
    root=BranchDefaults(n_layers=6, n_heads=8, reduction=Attention(n_outputs=16)),
    branch=BranchDefaults(n_layers=3, n_heads=8, reduction=Attention(n_outputs=8)),
    leaf=LeafDefaults(n_heads=8),
)
