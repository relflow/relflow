"""Generator illustrations for optional studies with no learning record."""

from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np

from evidence.diagnostics.plots import COLORS, axes_pair


def illustrate(identifier: str):
    if identifier == "P091":
        fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        boxes = [
            (0.05, 0.38, 0.2, 0.24, "8 observed\ncoordinates"),
            (0.38, 0.38, 0.22, 0.24, "Shared encoder\nwidth 6"),
            (0.77, 0.65, 0.2, 0.22, "Primary target\nfirst 4 coordinates"),
            (0.77, 0.15, 0.2, 0.22, "Auxiliary target\nlast 4 + noise"),
        ]
        for x, y, w, h, text in boxes:
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=COLORS[0], lw=1.8))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)
        for start, end in [((0.25, 0.5), (0.38, 0.5)), ((0.6, 0.5), (0.77, 0.76)), ((0.6, 0.5), (0.77, 0.26))]:
            ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": COLORS[2], "lw": 1.6})
        ax.text(
            0.49,
            0.12,
            "Single-task · Summed objectives · Jacobian objectives\nAuxiliary weights 1 and 8 are separate conditions",
            ha="center",
            va="center",
            fontsize=10,
        )
        return (
            fig,
            "Generator and architecture illustration. Eight independent observed coordinates feed a six-wide shared representation and two four-coordinate targets. Auxiliary noise has standard deviation 0.3. Objective interference remains unmeasured until this optional study runs.",
        )
    if identifier == "P092":
        fig, axes = axes_pair(width=10, height=4.8)
        matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        axes[0].imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axes[0].set(
            xticks=[0, 1],
            xticklabels=["Ordinary policy", "Inverted policy"],
            yticks=range(4),
            yticklabels=["Refund", "Cancel", "Pay", "Update profile"],
            title="The review rule needs both inputs",
        )
        for i, j in np.ndindex(matrix.shape):
            axes[0].text(
                j,
                i,
                "Review" if matrix[i, j] else "No review",
                ha="center",
                va="center",
                color="#18231e" if matrix[i, j] else "white",
                fontsize=11,
            )
        axes[1].barh([0, 1, 2], [4, 2, 3], color=[COLORS[0], COLORS[2], COLORS[1]])
        axes[1].set(
            yticks=[0, 1, 2],
            yticklabels=["Training", "Validation", "Held-out test"],
            xlabel="Paraphrase families per intent",
            title="Phrase families remain separate",
        )
        axes[1].invert_yaxis()
        return (
            fig,
            "Illustrated corpus design and answer rule, with no model predictions. Each of four intents has four training, two validation, and three held-out phrase families. A real pinned offline encoder is required to measure transfer.",
        )
    if identifier == "P099":
        fig, axes = axes_pair(width=10, height=4.6)
        elapsed = np.linspace(-10, 10, 101)
        axes[0].plot(elapsed, 2 * elapsed, color=COLORS[0], lw=2)
        axes[0].set(
            xlabel="Signed elapsed days (end minus start)",
            ylabel="Noiseless target",
            title="Generating rule: target = 2 × elapsed days",
        )
        axes[0].grid(alpha=0.18)
        axes[1].axis("off")
        start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        end = start + timedelta(days=1.5)
        for y, shift in [(0.72, 0), (0.34, 371)]:
            a, b = start + timedelta(days=shift), end + timedelta(days=shift)
            axes[1].text(
                0.05, y, a.strftime("%Y-%m-%d\n%H:%M UTC"), ha="left", va="center", transform=axes[1].transAxes
            )
            axes[1].text(
                0.95, y, b.strftime("%Y-%m-%d\n%H:%M UTC"), ha="right", va="center", transform=axes[1].transAxes
            )
            axes[1].annotate(
                "",
                xy=(0.72, y),
                xytext=(0.28, y),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "<->", "color": COLORS[0]},
            )
            axes[1].text(0.5, y + 0.08, "1.5 days → target 3", ha="center", transform=axes[1].transAxes, fontsize=10)
        axes[1].text(
            0.5,
            0.08,
            "Both endpoints shift by 371 days;\nthe elapsed feature stays the same.",
            ha="center",
            transform=axes[1].transAxes,
            fontsize=10,
        )
        return (
            fig,
            "Illustrated generating mean and exact endpoint-translation property. Learning records are unavailable. A matching wheel exporting Duration is required before its model-equivalence checks can be evaluated.",
        )
    if identifier == "P100":
        fig, axes = axes_pair(width=10, height=4.5)
        domains = [
            (np.arange(-9, 10, 2), np.arange(-8, 9, 2), "Integer ranks", None),
            ([0, 2, 4], [1, 3], "Declared ordered labels", ["Very low", "Low", "Medium", "High", "Very high"]),
        ]
        for ax, (seen, unseen, title, labels) in zip(axes, domains, strict=True):
            ax.scatter(seen, np.ones(len(seen)), color=COLORS[0], s=55, label="Training answers", marker="o")
            ax.scatter(
                unseen,
                np.zeros(len(unseen)),
                facecolors="none",
                edgecolors=COLORS[1],
                s=60,
                label="Withheld answers",
                marker="s",
            )
            ax.set(
                yticks=[0, 1],
                yticklabels=["Withheld", "Training"],
                xlabel="Generating rank / signal center",
                ylim=(-0.5, 1.5),
                title=title,
            )
            if labels:
                ax.set(xticks=range(5), xticklabels=labels)
                ax.tick_params(axis="x", rotation=25, labelsize=9)
            else:
                ax.set_xticks(range(-9, 10, 3))
            ax.grid(axis="x", alpha=0.18)
            ax.legend(fontsize=8)
        return (
            fig,
            "Illustrated split design. Inputs equal rank plus Gaussian noise with standard deviation 0.03. The plot identifies missing training answers and contains no learned predictions. A matching Ordinal wheel is required for the declared comparisons.",
        )
    raise ValueError(f"No generator illustration is authored for {identifier}")
