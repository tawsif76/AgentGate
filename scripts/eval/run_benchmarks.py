"""Run zero-shot benchmarks with AgentGate-Base-142M."""

import argparse
import json
from pathlib import Path

from lm_eval import evaluator
from lm_eval.utils import make_table

from scripts.eval.agentgate_adapter import AgentGateLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/bpe_32000.json")
    parser.add_argument("--tasks", default="hellaswag")
    parser.add_argument(
        "--limit", type=int, default=10, help="0 evaluates the complete task"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 or greater")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    model = AgentGateLM(args.checkpoint, args.tokenizer, args.device)

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=0,
        limit=args.limit or None,
        log_samples=True,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = "full" if args.limit == 0 else f"pilot_{args.limit}"
    path = output_dir / f"agentgate_{'_'.join(tasks)}_{tag}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(make_table(results))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
