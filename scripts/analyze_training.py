import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TARGET_STEPS = 20_599
TARGET_TOKENS = 2_699_952_128


def load_jsonl(path):
    events = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON at line {line_number}")

    return events


def get_events(events, event_type):
    return [event for event in events if event.get("event") == event_type]


def deduplicate_eval_events(events):
    """
    Keep the last eval event for each (step, eval_type) pair.

    This allows subset and full evaluations at the same optimizer
    step to both survive.
    """
    by_key = {}

    for event in events:
        step = event.get("step")
        eval_type = event.get("eval_type")

        if step is not None:
            by_key[(step, eval_type)] = event

    return sorted(
        by_key.values(),
        key=lambda event: (
            event["step"],
            event.get("eval_type", ""),
        ),
    )


def deduplicate_by_step(events):
    """
    Keeps the last event for each step.

    Useful when training was resumed and the log contains
    overlapping entries around a checkpoint boundary.
    """
    by_step = {}

    for event in events:
        step = event.get("step")

        if step is not None:
            by_step[step] = event

    return [by_step[step] for step in sorted(by_step)]


def safe_mean(values):
    values = [value for value in values if value is not None]

    if not values:
        return None

    return float(np.mean(values))


def safe_min(values):
    values = [value for value in values if value is not None]

    if not values:
        return None

    return float(np.min(values))


def safe_max(values):
    values = [value for value in values if value is not None]

    if not values:
        return None

    return float(np.max(values))


def save_csv(events, path):
    if not events:
        return

    keys = []

    for event in events:
        for key in event:
            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(events)


def nearest_train_loss(train_events, step):
    if not train_events:
        return None

    event = min(train_events, key=lambda x: abs(x["step"] - step))

    return event.get("train_loss")


def calculate_recent_val_slope(eval_events, last_n=5):
    """
    Linear trend of validation loss over the last N evaluations.

    Returned value is validation-loss change per 1000 training steps.

    Negative = validation loss was still improving.
    Positive = validation loss was worsening.
    """
    if len(eval_events) < 2:
        return None

    recent = eval_events[-last_n:]

    steps = np.array([event["step"] for event in recent], dtype=np.float64)
    losses = np.array([event["val_loss"] for event in recent], dtype=np.float64)

    slope = np.polyfit(steps, losses, 1)[0]

    return float(slope * 1000)


def plot_loss(train_events, eval_events, output_dir):
    plt.figure(figsize=(10, 6))

    if train_events:
        train_steps = [event["step"] for event in train_events]
        train_losses = [event["train_loss"] for event in train_events]

        plt.plot(
            train_steps,
            train_losses,
            label="Train loss",
            linewidth=1.2,
            alpha=0.75,
        )

    if eval_events:
        early_events = [
            event for event in eval_events if event.get("eval_type") == "early"
        ]
        subset_events = [
            event for event in eval_events if event.get("eval_type") == "subset"
        ]
        full_events = [
            event for event in eval_events if event.get("eval_type") == "full"
        ]

        if early_events:
            early_steps = [event["step"] for event in early_events]
            early_losses = [event["val_loss"] for event in early_events]

            plt.plot(
                early_steps,
                early_losses,
                marker="s",
                markersize=6,
                linestyle="--",
                linewidth=1.5,
                label="Early validation",
            )

        if subset_events:
            subset_steps = [event["step"] for event in subset_events]
            subset_losses = [event["val_loss"] for event in subset_events]

            plt.plot(
                subset_steps,
                subset_losses,
                marker="o",
                markersize=4,
                linewidth=2,
                label="Subset validation (2.048M tokens)",
            )

        if full_events:
            full_steps = [event["step"] for event in full_events]
            full_losses = [event["val_loss"] for event in full_events]

            plt.scatter(
                full_steps,
                full_losses,
                marker="*",
                s=120,
                label="Full validation (10.006M tokens)",
                zorder=5,
            )

    plt.xlabel("Optimizer step")
    plt.ylabel("Cross-entropy loss")
    plt.title("AgentGate Pretraining Loss")

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_learning_rate(train_events, output_dir):
    if not train_events:
        return

    steps = [event["step"] for event in train_events]
    lr = [event["lr"] for event in train_events]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, lr)

    plt.xlabel("Optimizer step")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate Schedule")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "learning_rate.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_gradients(train_events, output_dir):
    if not train_events:
        return

    steps = [event["step"] for event in train_events]
    grad_norm = [event["grad_norm"] for event in train_events]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, grad_norm)

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="Gradient clipping threshold",
    )

    plt.xlabel("Optimizer step")
    plt.ylabel("Gradient norm")
    plt.title("Gradient Norm")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "gradient_norm.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_clipping(train_events, output_dir):
    if not train_events:
        return

    steps = [event["step"] for event in train_events]
    clip_frac = [event["clip_frac"] for event in train_events]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, clip_frac)

    plt.xlabel("Optimizer step")
    plt.ylabel("Clipping fraction")
    plt.title("Gradient Clipping Fraction")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "clipping_fraction.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_throughput(train_events, output_dir):
    if not train_events:
        return

    steps = [event["step"] for event in train_events]
    throughput = [event["tokens_per_sec"] for event in train_events]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, throughput)

    plt.xlabel("Optimizer step")
    plt.ylabel("Tokens / second")
    plt.title("Training Throughput")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "throughput.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_mfu(train_events, output_dir):
    if not train_events:
        return

    steps = [event["step"] for event in train_events]
    mfu = [event["mfu"] * 100 for event in train_events]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, mfu)

    plt.xlabel("Optimizer step")
    plt.ylabel("MFU (%)")
    plt.title("Model FLOPs Utilization")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "mfu.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_validation_perplexity(eval_events, output_dir):
    if not eval_events:
        return

    steps = [event["step"] for event in eval_events]
    perplexity = [event["val_ppl"] for event in eval_events]

    plt.figure(figsize=(10, 5))
    plt.plot(
        steps,
        perplexity,
        marker="o",
        markersize=4,
    )

    plt.xlabel("Optimizer step")
    plt.ylabel("Validation perplexity")
    plt.title("Validation Perplexity")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "validation_perplexity.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_validation_vs_compute(eval_events, output_dir):
    """Plot validation loss against cumulative wall-GPU time.

    Wall-GPU hours are used because they represent the actual elapsed GPU
    resource consumed by the run, including evaluation overhead. Full
    validation points are highlighted separately from subset evaluations.
    """
    if not eval_events:
        return

    subset_events = [
        event
        for event in eval_events
        if event.get("eval_type") in {"subset", "early"}
        and event.get("wall_gpu_hours") is not None
        and event.get("val_loss") is not None
    ]

    full_events = [
        event
        for event in eval_events
        if event.get("eval_type") == "full"
        and event.get("wall_gpu_hours") is not None
        and event.get("val_loss") is not None
    ]

    if not subset_events and not full_events:
        return

    plt.figure(figsize=(10, 6))

    if subset_events:
        subset_hours = [event["wall_gpu_hours"] for event in subset_events]
        subset_losses = [event["val_loss"] for event in subset_events]

        plt.plot(
            subset_hours,
            subset_losses,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Subset validation (2.048M tokens)",
        )

    if full_events:
        full_hours = [event["wall_gpu_hours"] for event in full_events]
        full_losses = [event["val_loss"] for event in full_events]

        plt.scatter(
            full_hours,
            full_losses,
            marker="*",
            s=120,
            label="Full validation (10.006M tokens)",
            zorder=5,
        )

    plt.xlabel("Cumulative wall GPU time (hours)")
    plt.ylabel("Validation cross-entropy loss")
    plt.title("Validation Loss vs Cumulative GPU Compute")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "validation_loss_vs_compute.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_gpu_cost(train_events, output_dir):
    """Plot cumulative GPU cost across training steps."""
    cost_events = [
        event
        for event in train_events
        if event.get("cost_usd") is not None and event.get("step") is not None
    ]

    if not cost_events:
        return

    steps = [event["step"] for event in cost_events]
    costs = [event["cost_usd"] for event in cost_events]

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        steps,
        costs,
        color="#1f77b4",
        linewidth=2,
        label="GPU Cost ($)",
    )

    plt.title("GPU Cost vs. Training Step", fontsize=12)
    plt.xlabel("Optimizer step", fontsize=10)
    plt.ylabel("Cost (USD)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(
        output_dir / "gpu_cost.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def build_summary(events, train_events, eval_events):
    summary = {
        "target": {
            "steps": TARGET_STEPS,
            "tokens": TARGET_TOKENS,
        }
    }

    step0_events = get_events(events, "step0")

    if step0_events:
        summary["initial"] = {
            "loss": step0_events[0].get("loss"),
            "expected_random_loss": step0_events[0].get("expected_loss"),
        }

    if train_events:
        final_train = train_events[-1]

        summary["final_training"] = {
            "step": final_train.get("step"),
            "tokens_seen": final_train.get("tokens_seen"),
            "train_loss": final_train.get("train_loss"),
            "learning_rate": final_train.get("lr"),
            "grad_norm": final_train.get("grad_norm"),
            "clip_frac": final_train.get("clip_frac"),
            "tokens_per_sec": final_train.get("tokens_per_sec"),
            "mfu": final_train.get("mfu"),
            "peak_mem_gb": final_train.get("peak_mem_gb"),
            "training_compute_hours": final_train.get("training_compute_hours"),
            "wall_gpu_hours": final_train.get("wall_gpu_hours"),
            "cost_usd": final_train.get("cost_usd"),
        }

        summary["training_averages"] = {
            "tokens_per_sec": safe_mean(
                [event.get("tokens_per_sec") for event in train_events]
            ),
            "mfu": safe_mean([event.get("mfu") for event in train_events]),
            "step_time": safe_mean([event.get("step_time") for event in train_events]),
            "grad_norm": safe_mean([event.get("grad_norm") for event in train_events]),
            "clip_frac": safe_mean([event.get("clip_frac") for event in train_events]),
        }

        summary["training_extremes"] = {
            "max_grad_norm": safe_max(
                [event.get("grad_norm") for event in train_events]
            ),
            "max_peak_memory_gb": safe_max(
                [event.get("peak_mem_gb") for event in train_events]
            ),
            "min_train_loss": safe_min(
                [event.get("train_loss") for event in train_events]
            ),
        }

    if eval_events:
        final_eval = eval_events[-1]

        best_eval = min(
            eval_events,
            key=lambda event: event["val_loss"],
        )

        final_train_loss = nearest_train_loss(
            train_events,
            final_eval["step"],
        )

        generalization_gap = None

        if final_train_loss is not None:
            generalization_gap = final_eval["val_loss"] - final_train_loss

        summary["final_validation"] = {
            "step": final_eval.get("step"),
            "tokens_seen": final_eval.get("tokens_seen"),
            "eval_type": final_eval.get("eval_type"),
            "val_tokens": final_eval.get("val_tokens"),
            "val_loss": final_eval.get("val_loss"),
            "val_ppl": final_eval.get("val_ppl"),
            "nearest_train_loss": final_train_loss,
            "generalization_gap": generalization_gap,
        }

        summary["best_validation"] = {
            "step": best_eval.get("step"),
            "tokens_seen": best_eval.get("tokens_seen"),
            "val_loss": best_eval.get("val_loss"),
            "val_ppl": best_eval.get("val_ppl"),
        }

        summary["validation_trend"] = {
            "first_val_loss": eval_events[0].get("val_loss"),
            "final_val_loss": final_eval.get("val_loss"),
            "absolute_improvement": (
                eval_events[0]["val_loss"] - final_eval["val_loss"]
            ),
            "recent_slope_loss_per_1000_steps": (
                calculate_recent_val_slope(eval_events)
            ),
        }

    session_end_events = get_events(events, "session_end")

    if session_end_events:
        summary["last_session_end"] = session_end_events[-1]

    return summary


def print_summary(summary):
    print("\n" + "=" * 60)
    print("AGENTGATE PRETRAINING SUMMARY")
    print("=" * 60)

    initial = summary.get("initial")

    if initial:
        print(f"\nInitial loss:       {initial['loss']:.4f}")

    final_training = summary.get("final_training")

    if final_training:
        print("\nTraining")
        print("-" * 60)
        print(f"Final step:         {final_training['step']:,}")
        print(f"Tokens seen:        {final_training['tokens_seen']:,}")
        print(f"Final train loss:   {final_training['train_loss']:.4f}")
        print(f"Final LR:           {final_training['learning_rate']:.8f}")
        print(f"Final grad norm:    {final_training['grad_norm']:.4f}")

    final_validation = summary.get("final_validation")

    if final_validation:
        print("\nValidation")
        print("-" * 60)
        print(f"Final val loss:     {final_validation['val_loss']:.4f}")
        print(f"Final val PPL:      {final_validation['val_ppl']:.4f}")

        gap = final_validation.get("generalization_gap")

        if gap is not None:
            print(f"Generalization gap: {gap:+.4f}")

    best_validation = summary.get("best_validation")

    if best_validation:
        print("\nBest Validation")
        print("-" * 60)
        print(f"Step:               {best_validation['step']:,}")
        print(f"Loss:               {best_validation['val_loss']:.4f}")
        print(f"Perplexity:         {best_validation['val_ppl']:.4f}")

    trend = summary.get("validation_trend")

    if trend:
        slope = trend.get("recent_slope_loss_per_1000_steps")

        print("\nValidation Trend")
        print("-" * 60)
        print(
            f"First → final:      "
            f"{trend['first_val_loss']:.4f} → "
            f"{trend['final_val_loss']:.4f}"
        )
        print(f"Total improvement:  {trend['absolute_improvement']:.4f}")

        if slope is not None:
            print(f"Recent slope:        {slope:+.4f} loss / 1K steps")

    averages = summary.get("training_averages")
    final_training = summary.get("final_training")

    if averages and final_training:
        print("\nEfficiency")
        print("-" * 60)
        print(f"Average throughput: {averages['tokens_per_sec']:,.0f} tok/s")
        print(f"Average MFU:        {averages['mfu'] * 100:.2f}%")
        print(f"Peak GPU memory:    {final_training['peak_mem_gb']:.2f} GB")
        print(f"Compute hours:      {final_training['training_compute_hours']:.2f}")
        print(f"Wall GPU hours:     {final_training['wall_gpu_hours']:.2f}")
        print(f"Estimated cost:     ${final_training['cost_usd']:.2f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AgentGate pretraining JSONL logs."
    )

    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the training JSONL log.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_analysis"),
        help="Directory for analysis outputs.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = load_jsonl(args.log_file)

    train_events = deduplicate_by_step(get_events(events, "train"))
    eval_events = deduplicate_eval_events(get_events(events, "eval"))

    if not train_events:
        raise RuntimeError("No train events found in JSONL file.")

    if not eval_events:
        raise RuntimeError("No eval events found in JSONL file.")

    summary = build_summary(
        events,
        train_events,
        eval_events,
    )

    save_csv(
        train_events,
        args.output_dir / "train_metrics.csv",
    )

    save_csv(
        eval_events,
        args.output_dir / "eval_metrics.csv",
    )

    with open(
        args.output_dir / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    # Export all figures as vector PDFs
    plot_loss(
        train_events,
        eval_events,
        args.output_dir,
    )

    plot_learning_rate(
        train_events,
        args.output_dir,
    )

    plot_gradients(
        train_events,
        args.output_dir,
    )

    plot_clipping(
        train_events,
        args.output_dir,
    )

    plot_throughput(
        train_events,
        args.output_dir,
    )

    plot_mfu(
        train_events,
        args.output_dir,
    )

    plot_validation_perplexity(
        eval_events,
        args.output_dir,
    )

    plot_validation_vs_compute(
        eval_events,
        args.output_dir,
    )

    plot_gpu_cost(
        train_events,
        args.output_dir,
    )

    print_summary(summary)

    print(f"\nAnalysis saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
