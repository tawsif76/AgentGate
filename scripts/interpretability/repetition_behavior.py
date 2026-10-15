import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.model.agent_gate import AgentGate, AgentGateConfig


def load_model(path, device):
    model = AgentGate(AgentGateConfig())
    raw = torch.load(
        path, map_location="cpu", weights_only=False
    )  # trusted local checkpoint only
    state = raw
    if isinstance(raw, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(raw.get(key), dict):
                state = raw[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dict")
    state = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def token_nll(model, seq):
    # seq: [B, 2*N+1], first token acts as a prefix for the first prediction
    logits, _ = model(seq[:, :-1])
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        seq[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(seq.shape[0], -1)  # [B, 2*N]


def run_length(model, n, examples, batch_size, generator, device, low, high):
    first_rows, second_rows, control_rows = [], [], []
    for start in range(0, examples, batch_size):
        b = min(batch_size, examples - start)
        # Avoid special-token IDs by default; users can set the range manually.
        block = torch.randint(low, high, (b, n), generator=generator)
        prefix = torch.randint(low, high, (b, 1), generator=generator)
        positions = torch.stack(
            [torch.randperm(n, generator=generator) for _ in range(b)]
        )
        scrambled = block.gather(1, positions)
        original = torch.cat([prefix, block, block], dim=1).to(device)
        control = torch.cat([prefix, block, scrambled], dim=1).to(device)
        with torch.inference_mode():
            original_nll = token_nll(model, original).cpu().numpy()
            control_nll = token_nll(model, control).cpu().numpy()
        first_rows.append(original_nll[:, :n])
        second_rows.append(original_nll[:, n:])
        control_rows.append(control_nll[:, n:])
    first = np.concatenate(first_rows, axis=0)
    second = np.concatenate(second_rows, axis=0)
    control = np.concatenate(control_rows, axis=0)
    # Offset 0 in the second copy is not a valid next-token induction match;
    # compare offsets 1..N-1 to avoid claiming induction at the boundary.
    f = first[:, 1:].mean(axis=1)
    s = second[:, 1:].mean(axis=1)
    c = control[:, 1:].mean(axis=1)
    improvement = f - s
    specificity = c - s
    return (first, second, control), {
        "block_length": n,
        "examples": examples,
        "first_copy_nll": float(f.mean()),
        "second_copy_nll": float(s.mean()),
        "shuffled_control_second_nll": float(c.mean()),
        "copy_advantage_nats": float(improvement.mean()),
        "copy_advantage_sem": float(improvement.std(ddof=1) / np.sqrt(examples))
        if examples > 1
        else None,
        "control_gap_nats": float(specificity.mean()),
        "control_gap_sem": float(specificity.std(ddof=1) / np.sqrt(examples))
        if examples > 1
        else None,
        "second_better_fraction": float(np.mean(improvement > 0)),
    }


def _mean_and_sem(arr):
    """Compute position-wise mean and standard error across examples."""
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def plot_curves(curves, output, filename="repetition_loss_combined.pdf"):
    """Generate a publication-grade 1xK side-by-side vector PDF figure for arXiv."""
    # Configure Matplotlib rcParams for LaTeX / arXiv publication quality
    rc_Context = {
        "pdf.fonttype": 42,  # Embed TrueType fonts (avoids Type 3 font rejection)
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
    }

    # Okabe-Ito colorblind-safe palette + distinct line styles for grayscale printing
    styles = {
        "first": {
            "label": "First copy",
            "color": "#56B4E9",  # Sky blue
            "linestyle": "--",
            "zorder": 2,
        },
        "second": {
            "label": "Exact second copy",
            "color": "#0072B2",  # Deep blue
            "linestyle": "-",
            "zorder": 4,
        },
        "control": {
            "label": "Shuffled second copy",
            "color": "#D55E00",  # Vermilion / Orange-red
            "linestyle": ":",
            "zorder": 3,
        },
    }

    num_plots = len(curves)
    # Standard two-column width in LaTeX (e.g., NeurIPS/ICML/arXiv) is ~6.75 to 7.0 inches
    fig_width = max(7.0, 2.35 * num_plots)
    fig_height = 2.6

    with mpl.rc_context(rc_Context):
        fig, axes = plt.subplots(
            1,
            num_plots,
            figsize=(fig_width, fig_height),
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]

        for idx, (n, (first, second, control)) in enumerate(curves.items()):
            ax = axes[idx]
            x = np.arange(n)

            for key, data in [
                ("first", first),
                ("control", control),
                ("second", second),
            ]:
                mean, sem = _mean_and_sem(data)
                st = styles[key]
                ax.plot(
                    x,
                    mean,
                    label=st["label"],
                    color=st["color"],
                    linestyle=st["linestyle"],
                    zorder=st["zorder"],
                )
                if data.shape[0] > 1:
                    ax.fill_between(
                        x,
                        mean - sem,
                        mean + sem,
                        color=st["color"],
                        alpha=0.18,
                        linewidth=0,
                        zorder=st["zorder"] - 1,
                    )

            panel_letter = chr(ord("a") + idx)
            ax.set_title(f"({panel_letter}) Block length $N = {n}$", pad=6)
            ax.set_xlabel("Position within block")
            ax.set_xlim(0, n - 1)

            # Clean academic axis styling
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

        axes[0].set_ylabel("Next-token NLL (nats)")

        # Create a single shared legend at the top of the figure
        handles, labels = axes[0].get_legend_handles_labels()
        # Order legend: First copy, Exact second copy, Shuffled control
        order = [0, 2, 1]
        fig.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,
            frameon=False,
            handlelength=2.2,
            columnspacing=1.8,
        )

        fig.tight_layout()
        # Reserve space at the top for the shared legend
        fig.subplots_adjust(top=0.80, wspace=0.12)

        pdf_path = output / filename
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved publication figure to: {pdf_path.resolve()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--examples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-low", type=int, default=256)
    p.add_argument(
        "--token-high",
        type=int,
        default=32000,
        help="Exclusive; within tokenizer vocabulary",
    )
    p.add_argument("--output", default="interpretability_results/repetition")
    p.add_argument(
        "--figure-name",
        default="repetition_loss_combined.pdf",
        help="Filename for the combined PDF figure",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.examples < 1 or args.batch_size < 1 or min(args.lengths) < 2:
        p.error("Need examples >= 1, batch-size >= 1, and block lengths >= 2")
    if (
        args.token_low < 0
        or args.token_high > 32000
        or args.token_low >= args.token_high
    ):
        p.error("Token range must lie inside [0, 32000) and low < high")
    if 2 * max(args.lengths) + 1 > 2048:
        p.error("2 * max(lengths) + 1 exceeds ActionGate context length")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    model = load_model(args.checkpoint, args.device)
    curves = {}
    summaries = []
    for n in args.lengths:
        curves[n], summary = run_length(
            model,
            n,
            args.examples,
            args.batch_size,
            gen,
            args.device,
            args.token_low,
            args.token_high,
        )
        summaries.append(summary)
        print(
            f"N={n:3d} | first={summary['first_copy_nll']:.3f} | "
            f"repeat={summary['second_copy_nll']:.3f} | "
            f"shuffled={summary['shuffled_control_second_nll']:.3f} | "
            f"advantage={summary['copy_advantage_nats']:+.3f} nats | "
            f"control gap={summary['control_gap_nats']:+.3f} nats",
            flush=True,
        )
    plot_curves(curves, out, filename=args.figure_name)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    (out / "run_settings.json").write_text(json.dumps(vars(args), indent=2))
    print("Saved outputs to:", out.resolve())


if __name__ == "__main__":
    main()
