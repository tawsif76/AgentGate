#!/usr/bin/env python3
"""ActionGate: test whether earlier heads causally support later induction attention.

From repository root:
 python -m scripts.interpretability.head_dependency --checkpoint checkpoints/final.pt

Zero ablation of the PRE-output-projection per-head 64D vector; weights unchanged.
Measures the same induction-position attention score as head_specialization.py.
"""

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model.agent_gate import AgentGate, AgentGateConfig, MultiHeadAttention

# layer and head indices are zero-based
CASES = {
    "baseline": [],
    "previous_L16H2": [(16, 2)],
    "previous_L16H4": [(16, 4)],
    "previous_L16H2_H4": [(16, 2), (16, 4)],
    # Same-layer comparisons from Experiment 3
    "comparison_L16H0": [(16, 0)],
    "comparison_L16H7": [(16, 7)],
    "comparison_L16H0_H7": [(16, 0), (16, 7)],
    # Additional nearby-layer comparisons
    "comparison_L15H0": [(15, 0)],
    "comparison_L17H0": [(17, 0)],
}
TARGETS = [(19, 8), (19, 9), (18, 8)]


def load_model(checkpoint, device):
    config = AgentGateConfig()
    model = AgentGate(config)
    raw = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )  # trusted local checkpoint
    state = raw
    if isinstance(raw, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if isinstance(raw.get(k), dict):
                state = raw[k]
                break
    if not isinstance(state, dict):
        raise ValueError("No model state dict in checkpoint")
    state = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for block in model.blocks:
        block.attn.inspect_attention = True
    return model


@contextmanager
def ablate_heads(model, heads):
    """Ablate each head before W_O, leaving attention weights available for scoring."""
    by_layer = {}
    for layer, head in heads:
        if not (0 <= layer < len(model.blocks) and 0 <= head < model.config.n_heads):
            raise ValueError(f"Invalid layer/head {(layer, head)}")
        by_layer.setdefault(layer, []).append(head)
    handles = []
    try:
        for layer, selected in by_layer.items():
            n_heads = model.config.n_heads
            head_dim = model.config.head_dim

            def hook(
                _module,
                args,
                selected=tuple(selected),
                n_heads=n_heads,
                head_dim=head_dim,
            ):
                (flat,) = args
                b, t, c = flat.shape
                assert c == n_heads * head_dim
                x = flat.reshape(b, t, n_heads, head_dim).clone()
                x[:, :, list(selected), :] = 0
                return (x.reshape(b, t, c),)

            handles.append(
                model.blocks[layer].attn.proj.register_forward_pre_hook(hook)
            )
        yield
    finally:
        for h in handles:
            h.remove()


def compute_scores(model, x, N):
    """Return B x targets induction scores and B second-copy token NLL."""
    q = torch.arange(N + 1, 2 * N, device=x.device)
    k = q - N + 1
    logits, _ = model(x[:, :-1])
    values = []
    for layer, head in TARGETS:
        att = model.blocks[layer].attn.last_attn_weights
        if att is None:
            raise RuntimeError("Need inspect_attention=True on all attention modules")
        values.append(att[:, head, q, k].float().mean(-1))
    # Logits position (N .. 2N-1) predict second-copy tokens at (N+1 .. 2N).
    # Exclude the initial token, as in repetition_behavior.py.
    target_pos = torch.arange(N + 2, 2 * N + 1, device=x.device)
    pred_pos = target_pos - 1
    logprob = logits[:, pred_pos, :].float().log_softmax(-1)
    targets = x[:, target_pos]
    second_nll = -logprob.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean(-1)
    for block in model.blocks:
        block.attn.last_attn_weights = None
    return torch.stack(values, dim=1).cpu(), second_nll.cpu()


def _normalize_label(name):
    """Ensure any control_ prefix is converted to comparison_."""
    if name.startswith("control_"):
        return "comparison_" + name.removeprefix("control_")
    return name


def plot_dependency_effects(rows, pdf_path):
    """Generate a publication-grade grouped horizontal bar chart PDF for arXiv."""
    rc_context = {
        "pdf.fonttype": 42,  # Embed TrueType fonts (avoids Type 3 font rejection)
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
    }

    order = [name for name in CASES if name != "baseline"]
    display_labels = [_normalize_label(name) for name in order]

    # Add a visual gap between previous-token candidates and comparison heads
    y_positions = []
    current_y = 0.0
    divider_y = None
    for idx, name in enumerate(order):
        if (
            idx > 0
            and order[idx - 1].startswith("previous_")
            and not name.startswith("previous_")
        ):
            divider_y = current_y - 0.32
            current_y += 0.50
        y_positions.append(current_y)
        current_y += 1.0
    ys = np.array(y_positions)

    # Okabe-Ito colorblind-safe palette for the three downstream induction heads
    target_specs = [
        ("L19H8", -0.24, "#0072B2"),  # Deep blue
        ("L19H9", 0.00, "#D55E00"),  # Vermilion / Orange
        ("L18H8", +0.24, "#009E73"),  # Bluish green
    ]

    all_vals = []
    all_errs = []

    with mpl.rc_context(rc_context):
        fig, ax = plt.subplots(figsize=(6.8, 4.3))

        for target, offset, color in target_specs:
            selected = [
                next(
                    r
                    for r in rows
                    if r["intervention"] == name and r["target"] == target
                )
                for name in order
            ]
            vals = np.array([r["change_score"] for r in selected], dtype=float)
            errs = np.array([r["change_sem"] for r in selected], dtype=float)
            all_vals.append(vals)
            all_errs.append(errs)

            ax.barh(
                ys + offset,
                vals,
                height=0.22,
                label=f"Target: {target}",
                color=color,
                edgecolor="#1F2428",
                linewidth=0.5,
                zorder=3,
            )
            ax.errorbar(
                vals,
                ys + offset,
                xerr=errs,
                fmt="none",
                ecolor="#111111",
                elinewidth=0.9,
                capsize=2.0,
                capthick=0.8,
                zorder=4,
            )

        vals_flat = np.concatenate(all_vals)
        errs_flat = np.concatenate(all_errs)
        x_min = float(np.min(vals_flat - errs_flat)) - 0.04
        x_max = max(0.04, float(np.max(vals_flat + errs_flat)) + 0.03)
        ax.set_xlim(x_min, x_max)

        # Zero baseline reference line and group divider (without text annotations)
        ax.axvline(0, color="#333333", linewidth=0.85, linestyle="-", zorder=2)
        if divider_y is not None:
            ax.axhline(
                divider_y,
                color="#B0B8C0",
                linewidth=0.8,
                linestyle="--",
                zorder=1,
            )

        ax.set_yticks(ys)
        ax.set_yticklabels(display_labels)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Change in downstream induction-pattern score $\\Delta S_{\\mathrm{ind}} = S_{\\mathrm{ablated}} - S_{\\mathrm{normal}}$"
        )

        # Despine top and right axes for clean academic presentation
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.8,
        )

        fig.tight_layout()
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved publication dependency figure to: {pdf_path.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/final.pt")
    ap.add_argument("--examples", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--block-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument(
        "--output", type=Path, default=Path("interpretability_results/dependency")
    )
    ap.add_argument(
        "--figure-name",
        default="dependency_effects.pdf",
        help="Filename for the publication PDF figure",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.block_length < 3 or args.examples < 2 or args.batch_size < 1:
        ap.error("Need block-length >=3, examples >=2 and batch-size >=1")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print("Loading checkpoint...", flush=True)
    model = load_model(args.checkpoint, device)
    N = args.block_length
    if 2 * N > model.config.context_length:
        ap.error("2*block-length must not exceed model context")
    generator = torch.Generator().manual_seed(args.seed)
    blocks = torch.randint(
        256, model.config.vocab_size, (args.examples, N), generator=generator
    )
    prefix = torch.full((args.examples, 1), 1, dtype=torch.long)
    examples = torch.cat([prefix, blocks, blocks], dim=1)
    records = {name: {"scores": [], "nll": []} for name in CASES}
    with torch.inference_mode():
        for begin in range(0, args.examples, args.batch_size):
            batch = examples[begin : begin + args.batch_size].to(device)
            for name, heads in CASES.items():
                with ablate_heads(model, heads):
                    score, nll = compute_scores(model, batch, N)
                records[name]["scores"].append(score)
                records[name]["nll"].append(nll)
            print(
                f"Processed {min(begin + args.batch_size, args.examples)}/{args.examples}",
                flush=True,
            )
    baseline = torch.cat(records["baseline"]["scores"])
    baseline_nll = torch.cat(records["baseline"]["nll"])
    rows = []
    for name, item in records.items():
        s = torch.cat(item["scores"])
        nll = torch.cat(item["nll"])
        for col, (layer, head) in enumerate(TARGETS):
            change = s[:, col] - baseline[:, col]
            rows.append(
                dict(
                    intervention=_normalize_label(name),
                    target=f"L{layer}H{head}",
                    baseline_score=baseline[:, col].mean().item(),
                    ablated_score=s[:, col].mean().item(),
                    change_score=change.mean().item(),
                    change_sem=(change.std(unbiased=True) / len(change) ** 0.5).item(),
                    baseline_second_nll=baseline_nll.mean().item(),
                    ablated_second_nll=nll.mean().item(),
                    change_second_nll=(nll - baseline_nll).mean().item(),
                )
            )
    path = args.output / "dependency_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pdf_path = args.output / args.figure_name
    plot_dependency_effects(rows, pdf_path)

    (args.output / "run_settings.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output": str(args.output),
                "targets": TARGETS,
                "cases": CASES,
            },
            indent=2,
        )
    )
    print("\nResults:", path)
    for row in rows:
        if row["target"] in ("L19H8", "L19H9"):
            print(
                f"{row['intervention']:>22} -> {row['target']}: {row['baseline_score']:.3f} -> {row['ablated_score']:.3f} (Δ {row['change_score']:+.3f}); Δ second NLL {row['change_second_nll']:+.3f}"
            )
    print("Figure:", pdf_path)


if __name__ == "__main__":
    main()
