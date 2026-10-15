import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.model import AgentGate, AgentGateConfig


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
):
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    eot_id = tokenizer.token_to_id("<|endoftext|>")

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.context_length :]

        logits, _ = model(x_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

        if next_token.item() == eot_id:
            break

    return tokenizer.decode(
        x[0].tolist(),
        skip_special_tokens=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--tokenizer",
        default="artifacts/tokenizer/bpe_32000.json",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # -----------------------------
    # Tokenizer
    # -----------------------------

    tokenizer = Tokenizer.from_file(args.tokenizer)

    # -----------------------------
    # Model
    # -----------------------------

    config = AgentGateConfig()

    model = AgentGate(config)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    print(f"Parameters: {model.num_parameters():,}")

    # -----------------------------
    # Continuation prompts
    # -----------------------------

    prompts = [
        "The history of artificial intelligence began",
        "Computer networks allow multiple devices to",
        "The purpose of a firewall is to",
        "Machine learning is a field of computer science that",
        "In the future, autonomous vehicles may",
        "The scientist opened the laboratory door and",
        "Once upon a time, there was a small village",
        "The following Python function calculates the sum of two numbers:\n\ndef add(a, b):",
        "The weather was cold and the streets were",
        "A neural network learns by",
    ]
    output_dir = Path("generation_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sanity_generation_{timestamp}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, start=1):
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            result = (
                "\n" + "=" * 80 + "\n"
                f"PROMPT {i}\n" + "=" * 80 + "\n\n"
                f"Input:\n{prompt}\n\n"
                f"Generated:\n{output}\n"
            )

            print(result)
            f.write(result)

    print(f"\nSaved generations to: {output_file}")


if __name__ == "__main__":
    main()
