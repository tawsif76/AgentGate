import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model.agent_gate import AgentGate, AgentGateConfig, MultiHeadAttention


def load_model(checkpoint, device, toy=False):
    if toy:
        config = AgentGateConfig(
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=256,
            context_length=128,
        )
    else:
        config = AgentGateConfig()
    model = AgentGate(config)
    if checkpoint:
        raw = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )  # trusted checkpoint
        state = raw
        if isinstance(raw, dict):
            for name in ("model", "model_state_dict", "state_dict"):
                if isinstance(raw.get(name), dict):
                    state = raw[name]
                    break
        if not isinstance(state, dict):
            raise ValueError("No model state_dict found in checkpoint")
        state = {
            k.removeprefix("_orig_mod.").removeprefix("module."): v
            for k, v in state.items()
        }
        model.load_state_dict(state, strict=True)
        print(f"Loaded pretrained checkpoint: {checkpoint}", flush=True)
    else:
        print("WARNING: random toy model; not a pretrained result", flush=True)
    model = model.to(device).eval()
    attentions = [m for m in model.modules() if isinstance(m, MultiHeadAttention)]
    for m in attentions:
        m.inspect_attention = True
    return model, attentions


def score_batch(attentions, block_length, batch_size, device):
    """Return [layers, heads] scores averaged across batch and valid query positions.

    Full input is [prefix, block, block], length 2N+1.
    Previous-token queries: positions 1..(2N), keys q-1.
    Induction queries: positions (N+1)..(2N-1), keys (q-N+1).
    Why? At the second A at index N+1, attend to the earlier B at index 2
    (the token immediately AFTER the earlier A at index 1) to predict B.
    """
    N = block_length
    L = 2 * N + 1
    prev_q = torch.arange(1, L, device=device)
    ind_q = torch.arange(N + 1, 2 * N, device=device)  # N-1 valid queries
    prev_results = []
    induction_results = []
    for m in attentions:
        a = m.last_attn_weights
        if a is None or a.shape != (batch_size, m.n_heads, L, L):
            raise RuntimeError("Missing or incorrectly shaped attention tensor")
        # a : [batch, heads, query_position, key_position]
        p = a[:, :, prev_q, prev_q - 1].float().mean(dim=(0, 2))
        i = a[:, :, ind_q, ind_q - N + 1].float().mean(dim=(0, 2))
        prev_results.append(p.cpu())
        induction_results.append(i.cpu())
        # Let this layer release its full T x T attention tensor now.
        m.last_attn_weights = None
    return torch.stack(prev_results), torch.stack(induction_results)


def plot_combined_heatmaps(
    previous,
    induction,
    path,
    cmap="viridis",
    annotate_threshold=0.50,
):
    """Generate a publication-grade 1x2 side-by-side heatmap vector PDF for arXiv."""
    rc_context = {
        "pdf.fonttype": 42,  # Embed TrueType fonts for arXiv / LaTeX
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    }

    n_layers, n_heads = previous.shape
    # Scale height gracefully for both 2-layer toy models and 24-layer full models
    fig_height = max(3.2, min(5.4, 0.20 * n_layers + 0.8))
    fig_width = 7.0  # Standard two-column LaTeX textwidth

    panels = [
        ("(a) Previous-token attention ($S_{\\mathrm{prev}}$)", previous),
        ("(b) Induction-pattern attention ($S_{\\mathrm{ind}}$)", induction),
    ]

    with mpl.rc_context(rc_context):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(fig_width, fig_height),
            gridspec_kw={"width_ratios": [1, 1, 0.045], "wspace": 0.14},
        )
        ax_prev, ax_ind, cax = axes

        for idx, (ax, (title, matrix)) in enumerate(zip((ax_prev, ax_ind), panels)):
            im = ax.imshow(
                matrix,
                aspect="auto",
                origin="upper",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )

            # Major ticks for layer & head indices
            ax.set_xticks(np.arange(n_heads))
            ax.set_xticklabels(np.arange(n_heads))
            ax.set_yticks(np.arange(n_layers))
            if idx == 0:
                ax.set_yticklabels(np.arange(n_layers))
                ax.set_ylabel("Transformer layer (0-indexed)")
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            ax.set_xlabel("Attention head (0-indexed)")
            ax.set_title(title, pad=8)

            # Subtle gridlines between cells for crisp visual separation
            ax.set_xticks(np.arange(-0.5, n_heads, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_layers, 1), minor=True)
            ax.grid(
                which="minor",
                color="white",
                linestyle="-",
                linewidth=0.35,
                alpha=0.28,
            )
            ax.tick_params(which="minor", bottom=False, left=False)

            # Annotate salient cells above threshold (e.g., L16H2, L16H4, L19H8, L19H9)
            if annotate_threshold is not None and annotate_threshold < 1.0:
                for l in range(n_layers):
                    for h in range(n_heads):
                        val = matrix[l, h]
                        if val >= annotate_threshold:
                            text_color = "#111111" if val >= 0.62 else "white"
                            ax.text(
                                h,
                                l,
                                f"{val:.2f}".lstrip("0"),
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=6.8,
                                fontweight="bold",
                            )

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Mean attention mass on designated key", labelpad=6)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_linewidth(0.6)

        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined publication heatmap to: {path.resolve()}", flush=True)


def save_top(rows, key, path, topk):
    ranked = sorted(rows, key=lambda r: r[key], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "layer", "head", "score"])
        w.writeheader()
        for rank, r in enumerate(ranked[:topk], 1):
            w.writerow(
                {"rank": rank, "layer": r["layer"], "head": r["head"], "score": r[key]}
            )
    print(f"\nTop {min(topk, len(ranked))} by {key}:", flush=True)
    for r in ranked[:topk]:
        print(f"  L{r['layer']:02d} H{r['head']:02d}: {r[key]:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--block-length", type=int, default=64)
    p.add_argument("--examples", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--token-low", type=int, default=256)
    p.add_argument(
        "--token-high", type=int, default=32000, help="Exclusive upper bound"
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument(
        "--output", type=Path, default=Path("interpretability_results/heads")
    )
    p.add_argument(
        "--figure-name",
        default="head_specialization_combined.pdf",
        help="Filename for the combined PDF heatmap figure",
    )
    p.add_argument(
        "--annotate-threshold",
        type=float,
        default=0.50,
        help="Annotate cells with score >= threshold (set >= 1.0 to disable)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--toy", action="store_true", help="Quick random 2-layer model sanity test"
    )
    args = p.parse_args()
    if args.toy and args.checkpoint:
        p.error("--toy and --checkpoint cannot be combined")
    if not args.toy and not args.checkpoint:
        p.error("Provide --checkpoint for real evaluation or --toy for sanity test")
    if (
        args.block_length < 2
        or args.examples < 1
        or args.batch_size < 1
        or args.topk < 1
    ):
        p.error("Block length >=2 and examples/batch-size/topk >=1 required")
    torch.manual_seed(args.seed)
    model, attentions = load_model(args.checkpoint, args.device, args.toy)
    if (
        args.token_low < 0
        or args.token_low >= args.token_high
        or args.token_high > model.config.vocab_size
    ):
        p.error("Invalid token ID range for model vocabulary")
    if 2 * args.block_length + 1 > model.config.context_length:
        p.error("2*block_length+1 exceeds model context length")
    gen = torch.Generator().manual_seed(args.seed)
    total_prev = torch.zeros(
        (model.config.n_layers, model.config.n_heads), dtype=torch.float64
    )
    total_ind = torch.zeros_like(total_prev)
    N = args.block_length
    for start in range(0, args.examples, args.batch_size):
        b = min(args.batch_size, args.examples - start)
        block = torch.randint(args.token_low, args.token_high, (b, N), generator=gen)
        prefix = torch.randint(args.token_low, args.token_high, (b, 1), generator=gen)
        seq = torch.cat([prefix, block, block], dim=1).to(args.device)
        with torch.inference_mode():
            model(seq)
            prev, ind = score_batch(attentions, N, b, seq.device)
        total_prev += prev.double() * b
        total_ind += ind.double() * b
        if (start // args.batch_size + 1) % 5 == 0 or start + b == args.examples:
            print(f"Processed {start + b}/{args.examples} examples", flush=True)
    previous = (total_prev / args.examples).numpy()
    induction = (total_ind / args.examples).numpy()
    args.output.mkdir(parents=True, exist_ok=True)

    plot_combined_heatmaps(
        previous,
        induction,
        args.output / args.figure_name,
        annotate_threshold=args.annotate_threshold,
    )

    rows = [
        {
            "layer": l,
            "head": h,
            "previous_token_score": float(previous[l, h]),
            "induction_score": float(induction[l, h]),
        }
        for l in range(model.config.n_layers)
        for h in range(model.config.n_heads)
    ]
    with (args.output / "all_head_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=["layer", "head", "previous_token_score", "induction_score"]
        )
        w.writeheader()
        w.writerows(rows)
    save_top(
        rows, "previous_token_score", args.output / "top_previous_heads.csv", args.topk
    )
    save_top(
        rows, "induction_score", args.output / "top_induction_heads.csv", args.topk
    )
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "run_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    for m in attentions:
        m.inspect_attention = False
        m.last_attn_weights = None
    print(f"\nSaved results to {args.output.resolve()}", flush=True)
    print(
        "NOTE: High attention-pattern scores are candidate evidence, NOT proof of a causal copying circuit."
    )


if __name__ == "__main__":
    main()
