"""Plot the recorded longer-training trajectories without rerunning models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent
SEEDS = (7801, 7802, 7803)
COLORS = ("#16816c", "#587fbd", "#ba7448")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for seed, color in zip(SEEDS, COLORS, strict=True):
        record = json.loads((ROOT / "output" / f"seed-{seed}-steps-4800.json").read_text())
        assert record["budget"] == 4800 and record["seed"] == seed
        arms = record["metrics"]["group_counts"]["4"]
        for arm, style in (("stable", "-"), ("no_regime", "--")):
            points = arms[arm]["ari_by_step"]
            assert points and all(point["assigned_entities"] == 64 for point in points)
            x = [point["step"] for point in points]
            label = f"Seed {seed}" if arm == "stable" else "No-regime control" if seed == SEEDS[0] else "_nolegend_"
            for axis, metric in zip(axes, ("adjusted_rand", "validation_rmse"), strict=True):
                axis.plot(
                    x,
                    [point[metric] for point in points],
                    color=color if arm == "stable" else "#9ca8a1",
                    linestyle=style,
                    alpha=1 if arm == "stable" else 0.65,
                    linewidth=2 if arm == "stable" else 1.3,
                    label=label,
                )
    axes[0].axhline(0.95, color="#a8524e", linestyle=":", label="Final test ARI target: 0.95")
    axes[0].set(
        ylabel="Assignment agreement (ARI)", ylim=(-0.05, 1.04), title="Do identities recover the hidden rules?"
    )
    axes[1].set(ylabel="Validation RMSE (target units)", title="Does predictive error improve?")
    for axis in axes:
        axis.axvline(600, color="#53665c", linestyle=":", linewidth=1.2, label="Original 600-update budget")
        axis.set_xscale("log")
        axis.set_xticks([32, 128, 600, 1200, 2400, 4800])
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.set_xlabel("Optimizer updates (log scale)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8, loc="best")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Group training duration: three exploratory seeds", fontsize=14)
    fig.savefig(ROOT / "training-duration.svg", bbox_inches="tight")
    fig.savefig(ROOT / "training-duration.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
