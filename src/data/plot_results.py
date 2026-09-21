import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_PATH = Path("artifacts/tokenizer/comparison_results.json")
PLOTS_DIR = Path("artifacts/tokenizer/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Clean, modern styling
COLOR_24K = "#2563EB"
COLOR_32K = "#6366F1"


def apply_minimal_style(ax, title, ylabel, ylim_max):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15, color="#111827")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", color="#374151")
    ax.set_ylim(0, ylim_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#9CA3AF")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11, labelcolor="#111827")
    ax.tick_params(axis="y", labelsize=10, labelcolor="#4B5563")


def plot_metrics(data):
    results = data["results"]
    labels = [f"{r['vocab_size'] // 1000}k Vocab" for r in results]

    # 1. Characters per Token (Compression)
    cpt = [r["characters_per_token"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bars = ax.bar(labels, cpt, color=[COLOR_24K, COLOR_32K], width=0.45, zorder=3)
    apply_minimal_style(
        ax,
        "Compression: Characters / Token\n(Higher is Better)",
        "Chars per Token",
        max(cpt) * 1.18,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.08,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    diff = ((cpt[1] - cpt[0]) / cpt[0]) * 100
    ax.text(
        0.5,
        max(cpt) * 1.10,
        f"+{diff:.2f}% compression gain",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#047857",
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=1
        ),
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "1_compression_chars_per_token.png")
    plt.close()

    # 2. Unseen Tokens % (Dead Weight)
    unseen = [r["unseen_percent"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bars = ax.bar(labels, unseen, color=[COLOR_24K, COLOR_32K], width=0.45, zorder=3)
    apply_minimal_style(
        ax,
        "Dead Weight: Unseen Tokens in Eval\n(Lower is Better)",
        "% of Vocabulary",
        max(unseen) * 1.25,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "2_dead_weight_unseen_tokens.png")
    plt.close()

    # 3. Rare Tokens % (Fragile Tail: 1-9 occurrences)
    rare = [r["rare_percent_1_to_9"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bars = ax.bar(labels, rare, color=[COLOR_24K, COLOR_32K], width=0.45, zorder=3)
    apply_minimal_style(
        ax,
        "Fragile Tail: Seen 1–9 Times in Eval\n(Lower is Better)",
        "% of Vocabulary",
        max(rare) * 1.25,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.05,
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "3_fragile_tail_tokens.png")
    plt.close()

    print(f"✓ All 3 clean charts saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    plot_metrics(data)
