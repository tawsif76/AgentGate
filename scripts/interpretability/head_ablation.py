#!/usr/bin/env python3
"""AgentGate Day 2: causal attention-head zero ablation (no training).

Run from project root:
 python -m scripts.interpretability.head_ablation --checkpoint checkpoints/final.pt --examples 100

Ablates each selected head's pre-output-projection 64-d contribution using a
forward PRE-hook on blocks[layer].attn.proj. Uses fused attention throughout.
Reports paired changes in second-copy NLL and in repetition advantage.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.model.agent_gate import AgentGate, AgentGateConfig


def load_model(path, device):
    model = AgentGate(AgentGateConfig())
    raw = torch.load(
        path, map_location="cpu", weights_only=False
    )  # trusted local checkpoint
    state = raw
    if isinstance(raw, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(raw.get(key), dict):
                state = raw[key]
                break
    if not isinstance(state, dict):
        raise ValueError("No model state dictionary found")
    state = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for blk in model.blocks:
        blk.attn.inspect_attention = False
    return model


def make_intervention(model, heads):
    """Zero heads before their concatenated output is passed through W_O."""
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    handles = []
    for layer, indices in by_layer.items():
        module = model.blocks[layer].attn
        dims = module.head_dim
        head_count = module.n_heads
        if any(h < 0 or h >= head_count for h in indices):
            raise ValueError(f"Invalid head index in layer {layer}: {indices}")

        def hook(_module, inputs, indices=tuple(indices), dims=dims):
            x = inputs[0]
            y = x.clone()
            for h in indices:
                y[..., h * dims : (h + 1) * dims] = 0
            return (y,)

        handles.append(module.proj.register_forward_pre_hook(hook))
    return handles


@torch.inference_mode()
def per_example_losses(model, seq, n):
    logits, _ = model(seq[:, :-1])
    targets = seq[:, 1:]
    nll = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(seq.shape[0], -1)
    # Exclude within-block index zero consistently for first and second copies.
    return nll[:, 1:n].mean(1), nll[:, n + 1 : 2 * n].mean(1)


def run_condition(model, seq_batches, n, heads):
    handles = make_intervention(model, heads)
    try:
        first, second = [], []
        for seq in seq_batches:
            f, s = per_example_losses(model, seq, n)
            first.append(f.cpu().numpy())
            second.append(s.cpu().numpy())
        return np.concatenate(first), np.concatenate(second)
    finally:
        for handle in handles:
            handle.remove()


def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None


def _format_condition_meta(cond):
    """Return (display_label, color, hatch) for publication plotting."""
    # Okabe-Ito colorblind-safe palette
    c_ind = "#0072B2"  # Deep blue (Induction candidates)
    c_prev = "#D55E00"  # Vermilion (Previous-token candidates)
    c_comp = "#8C969E"  # Neutral slate (Comparison heads)

    if cond.startswith("induction_"):
        return (
            cond,
            c_ind,
            "///"
            if cond.count("H") > 1 or "_" in cond.removeprefix("induction_")
            else "",
        )

    if cond.startswith("previous_"):
        return (
            cond,
            c_prev,
            "///"
            if cond.count("H") > 1 or "_" in cond.removeprefix("previous_")
            else "",
        )

    if cond.startswith("control_"):
        cond = "comparison_" + cond.removeprefix("control_")

    return cond, c_comp, ""


def plot_ablation_effects(rows, pdf_path):
    """Generate a publication-grade horizontal bar chart PDF for arXiv."""
    rc_context = {
        "pdf.fonttype": 42,  # Embed TrueType fonts (avoids Type 3 rejection)
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "hatch.linewidth": 0.6,
    }

    shown = [r for r in rows if r["condition"] != "baseline"]

    # Add a small vertical gap between the L18-19 group and L16 group
    y_positions = []
    current_y = 0.0
    divider_y = None
    for idx, r in enumerate(shown):
        if (
            idx > 0
            and shown[idx - 1]["condition"].endswith("L18H1")
            and r["condition"].startswith("previous_")
        ):
            divider_y = current_y - 0.35
            current_y += 0.55
        y_positions.append(current_y)
        current_y += 1.0
    y = np.array(y_positions)

    values = np.array([r["copy_advantage_drop"] for r in shown], dtype=float)
    errors = np.array([r["copy_advantage_drop_sem"] or 0.0 for r in shown], dtype=float)
    meta = [_format_condition_meta(r["condition"]) for r in shown]
    labels = [m[0] for m in meta]
    colors = [m[1] for m in meta]
    hatches = [m[2] for m in meta]

    with mpl.rc_context(rc_context):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))

        bars = ax.barh(
            y,
            values,
            height=0.68,
            color=colors,
            edgecolor="#1F2428",
            linewidth=0.6,
            zorder=3,
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)

        # Error bars (without text value labels)
        ax.errorbar(
            values,
            y,
            xerr=errors,
            fmt="none",
            ecolor="#111111",
            elinewidth=1.0,
            capsize=2.5,
            capthick=0.9,
            zorder=4,
        )

        x_min = min(-0.35, float(np.min(values - errors)) - 0.2)
        x_max = float(np.max(values + errors)) + 0.35
        ax.set_xlim(x_min, x_max)

        # Reference zero line and section divider
        ax.axvline(0, color="#333333", linewidth=0.85, linestyle="-", zorder=2)
        if divider_y is not None:
            ax.axhline(
                divider_y,
                color="#B0B8C0",
                linewidth=0.8,
                linestyle="--",
                zorder=1,
            )
            ax.text(
                x_max - 0.08,
                divider_y - 0.35,
                "Later layers (L18–L19: Induction stage)",
                ha="right",
                va="bottom",
                fontsize=7.8,
                fontstyle="italic",
                color="#4A525A",
            )
            ax.text(
                x_max - 0.08,
                divider_y + 0.35,
                "Earlier layer (L16: Previous-token stage)",
                ha="right",
                va="top",
                fontsize=7.8,
                fontstyle="italic",
                color="#4A525A",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Reduction in copy advantage $\\Delta A = A_{\\mathrm{normal}} - A_{\\mathrm{ablated}}$ (nats; positive = worse copying)"
        )

        # Clean academic axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

        # Custom legend at the top
        legend_handles = [
            mpatches.Patch(
                facecolor="#0072B2",
                edgecolor="#1F2428",
                linewidth=0.6,
                label="Induction candidate",
            ),
            mpatches.Patch(
                facecolor="#D55E00",
                edgecolor="#1F2428",
                linewidth=0.6,
                label="Previous-token candidate",
            ),
            mpatches.Patch(
                facecolor="white",
                edgecolor="#1F2428",
                hatch="///",
                linewidth=0.6,
                label="Combined 2-head ablation",
            ),
            mpatches.Patch(
                facecolor="#8C969E",
                edgecolor="#1F2428",
                linewidth=0.6,
                label="Comparison head",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.2,
        )

        fig.tight_layout()
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved publication ablation figure to: {pdf_path.resolve()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--block-length", type=int, default=64)
    p.add_argument("--examples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--token-low", type=int, default=256)
    p.add_argument("--token-high", type=int, default=32000)
    p.add_argument("--output", default="interpretability_results/ablation")
    p.add_argument(
        "--figure-name",
        default="ablation_effects.pdf",
        help="Filename for the publication PDF figure",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.examples < 2 or args.batch_size < 1 or args.block_length < 2:
        p.error("Requires examples >=2, batch-size >=1, block-length >=2")
    if (
        args.token_low < 0
        or args.token_high > 32000
        or args.token_low >= args.token_high
    ):
        p.error("Token range must be within vocabulary")
    if 2 * args.block_length + 1 > 2048:
        p.error("Sequence exceeds context length")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device)
    gen = torch.Generator().manual_seed(args.seed)
    batches = []
    for start in range(0, args.examples, args.batch_size):
        b = min(args.batch_size, args.examples - start)
        block = torch.randint(
            args.token_low, args.token_high, (b, args.block_length), generator=gen
        )
        prefix = torch.randint(args.token_low, args.token_high, (b, 1), generator=gen)
        batches.append(torch.cat([prefix, block, block], dim=1).to(args.device))

    # Fixed, same-layer head comparisons rather than cherry-picked unrelated layers.
    conditions = [
        ("baseline", []),
        ("induction_L19H8", [(19, 8)]),
        ("induction_L19H9", [(19, 9)]),
        ("induction_L18H8", [(18, 8)]),
        ("induction_L19H8_H9", [(19, 8), (19, 9)]),
        ("comparison_L19H0", [(19, 0)]),
        ("comparison_L19H3", [(19, 3)]),
        ("comparison_L18H1", [(18, 1)]),
        ("previous_L16H2", [(16, 2)]),
        ("previous_L16H4", [(16, 4)]),
        ("previous_L16H2_H4", [(16, 2), (16, 4)]),
        ("comparison_L16H0", [(16, 0)]),
        ("comparison_L16H7", [(16, 7)]),
    ]
    baseline_first, baseline_second = run_condition(
        model, batches, args.block_length, []
    )
    base_advantage = baseline_first - baseline_second
    print(
        f"Baseline: first={baseline_first.mean():.4f} second={baseline_second.mean():.4f} advantage={base_advantage.mean():.4f}"
    )
    rows = []
    for label, heads in conditions:
        if label == "baseline":
            first, second = baseline_first, baseline_second
        else:
            first, second = run_condition(model, batches, args.block_length, heads)
        # Positive means intervention degrades copying compared with the baseline.
        paired_second_change = second - baseline_second
        paired_advantage_drop = base_advantage - (first - second)
        rows.append(
            {
                "condition": label,
                "ablated_heads": ";".join(f"{l}.{h}" for l, h in heads),
                "first_nll": float(first.mean()),
                "second_nll": float(second.mean()),
                "copy_advantage": float((first - second).mean()),
                "delta_second_nll": float(paired_second_change.mean()),
                "delta_second_nll_sem": sem(paired_second_change),
                "copy_advantage_drop": float(paired_advantage_drop.mean()),
                "copy_advantage_drop_sem": sem(paired_advantage_drop),
            }
        )
        print(
            f"{label:25s} delta_second={rows[-1]['delta_second_nll']:+.4f} "
            f"delta_copy_advantage={rows[-1]['copy_advantage_drop']:+.4f} nats"
        )

    with (out / "ablation_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "run_settings.json").write_text(
        json.dumps(
            {**vars(args), "conditions": {k: v for k, v in conditions}}, indent=2
        )
    )

    plot_ablation_effects(rows, out / args.figure_name)
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
