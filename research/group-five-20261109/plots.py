"""Render seed-level lab readouts from the captured full-budget records."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
COLORS = ("#167d68", "#c87535", "#5c81b8")
PANELS = {
    "P102": (
        "Rare and frequent identities",
        [
            ("Group / rare", ("group", "rare")),
            ("Category / rare", ("category", "rare")),
            ("No identity / rare", ("no_identity", "rare")),
            ("Shuffled / rare", ("shuffled", "rare")),
            ("Group / frequent", ("group", "frequent")),
            ("Category / frequent", ("category", "frequent")),
        ],
    ),
    "P103": (
        "Number of group slots",
        [
            ("2 slots", ("two", "test")),
            ("4 slots", ("four", "test")),
            ("8 slots", ("eight", "test")),
            ("4 / shuffled", ("shuffled", "test")),
        ],
    ),
    "P104": (
        "Known, unseen, and null identities",
        [
            ("Known / no dropout", ("no_dropout", "known")),
            ("Unseen / no dropout", ("no_dropout", "unseen")),
            ("Null / no dropout", ("no_dropout", "null")),
            ("Known / dropout .10", ("dropout", "known")),
            ("Unseen / dropout .10", ("dropout", "unseen")),
            ("Null / dropout .10", ("dropout", "null")),
        ],
    ),
    "P105": (
        "Temporarily moving identities to another slot",
        [("Original assignment", ("original",)), ("Rotated assignment", ("intervention",))],
    ),
    "P106": (
        "Changed identities after behavior drift",
        [
            ("Before / changed identities", ("drift", "before", "changed")),
            ("After drift training / changed", ("drift", "after", "changed")),
            ("After stable training / changed", ("stable_control", "after", "changed")),
            ("Before / unchanged identities", ("drift", "before", "unchanged")),
            ("After drift training / unchanged", ("drift", "after", "unchanged")),
            ("After stable training / unchanged", ("stable_control", "after", "unchanged")),
        ],
    ),
    "P107": (
        "Fresh-seed confirmation of behavior drift",
        [
            ("Before / changed identities", ("drift", "before", "changed")),
            ("After drift training / changed", ("drift", "after", "changed")),
            ("After stable training / changed", ("stable_control", "after", "changed")),
            ("Before / unchanged identities", ("drift", "before", "unchanged")),
            ("After drift training / unchanged", ("drift", "after", "unchanged")),
            ("After stable training / unchanged", ("stable_control", "after", "unchanged")),
        ],
    ),
}


def main():
    target = ROOT / "plots"
    target.mkdir(exist_ok=True)
    for proof, (title, conditions) in PANELS.items():
        records = sorted(OUTPUT.glob(f"{proof}-*-full/record.json"))
        if len(records) != 3:
            raise ValueError(f"{proof}: expected three captured records, found {len(records)}")
        fig, ax = plt.subplots(figsize=(9, max(3.6, 0.55 * len(conditions) + 1.8)), layout="constrained")
        for index, path in enumerate(records):
            record = json.loads(path.read_text())
            values = []
            for _, keys in conditions:
                value = record["metrics"]
                for key in keys:
                    value = value[key]
                values.append(value["normalized_rmse"])
            offset = (index - 1) * 0.14
            ax.scatter(
                values,
                np.arange(len(values)) + offset,
                s=35,
                color=COLORS[index],
                label=f"Seed {record['seed']}",
                zorder=3,
            )
        ax.axvline(1, color="#a8524e", linestyle="--", label="Predict training mean")
        ax.set_yticks(range(len(conditions)), [name for name, _ in conditions])
        ax.invert_yaxis()
        ax.set_xlabel("Held-out normalized RMSE (lower is better)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.savefig(target / f"{proof}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
