#!/usr/bin/env python3
"""Plot AgentGate and reference-model zero-shot accuracy as vector PDFs."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASKS = ["ARC-Easy", "HellaSwag", "PIQA"]
DATA = {
    "AgentGate-142M": {"acc": [50.46, 29.14, 62.95], "acc_norm": [45.41, 32.03, 62.40]},
    "Pythia-160M": {"acc": [43.64, 28.37, 62.30], "acc_norm": [39.65, 30.27, 61.92]},
    "GPT-2 Small": {"acc": [43.81, 28.92, 62.89], "acc_norm": [39.48, 31.14, 62.51]},
    "SmolLM-135M": {"acc": [61.62, 35.18, 68.61], "acc_norm": [56.48, 42.60, 68.12]},
}


def plot_metric(metric: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    positions = np.arange(len(TASKS))
    width = 0.19
    offsets = (np.arange(len(DATA)) - (len(DATA) - 1) / 2) * width

    for offset, (model, metrics) in zip(offsets, DATA.items()):
        values = metrics[metric]
        bars = ax.bar(positions + offset, values, width=width, label=model)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values],
                     padding=3, fontsize=8, rotation=90)

    ax.set_xticks(positions, TASKS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 82)
    ax.set_title("Zero-shot accuracy" if metric == "acc" else "Zero-shot length-normalized accuracy")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper left", frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output_file = output_dir / f"accuracy_{metric}.pdf"
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    output_dir = Path("benchmark_results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("acc", "acc_norm"):
        plot_metric(metric, output_dir)
