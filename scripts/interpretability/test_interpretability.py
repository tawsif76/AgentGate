"""Compare fused and manual attention on an existing ActionGate checkpoint.

Example: python test_interpretability.py --checkpoint /path/to/checkpoint.pt
Without --checkpoint, this verifies implementation equivalence using random weights only.
"""

import argparse

import torch

from src.model.agent_gate import AgentGate, AgentGateConfig, MultiHeadAttention


def set_inspection(model, enabled):
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            m.inspect_attention = enabled
            if not enabled:
                m.last_attn_weights = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--toy", action="store_true", help="Small random model for quick CPU test"
    )
    args = p.parse_args()
    if args.toy and args.checkpoint:
        p.error("--toy cannot be combined with --checkpoint")

    config = (
        AgentGateConfig(
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=256,
            context_length=128,
        )
        if args.toy
        else AgentGateConfig()
    )
    model = AgentGate(config)
    if args.checkpoint:
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = raw
        if isinstance(raw, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if key in raw and isinstance(raw[key], dict):
                    state = raw[key]
                    break
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a state dictionary")
        # Some DDP/compiled model checkpoints prefix parameter names.
        state = {
            k.removeprefix("_orig_mod.").removeprefix("module."): v
            for k, v in state.items()
        }
        model.load_state_dict(state, strict=True)
        print("Loaded pretrained checkpoint:", args.checkpoint)
    else:
        print("WARNING: No pretrained checkpoint supplied; testing RANDOM weights.")
    model = model.to(args.device).eval()
    torch.manual_seed(42)
    x = torch.randint(0, config.vocab_size, (1, args.seq_len), device=args.device)
    with torch.inference_mode():
        set_inspection(model, False)
        fused, _ = model(x)
        set_inspection(model, True)
        manual, _ = model(x)
        attention = [block.attn.last_attn_weights for block in model.blocks]
        differences = (fused.float() - manual.float()).abs()
        print("Max absolute logit difference:", differences.max().item())
        print("Mean absolute logit difference:", differences.mean().item())
        print(
            "Same argmax predictions:", torch.equal(fused.argmax(-1), manual.argmax(-1))
        )
        print("Captured attention shapes:", [tuple(a.shape) for a in attention[:2]])
        print(
            "Max row-sum error:",
            max((a.float().sum(-1) - 1).abs().max().item() for a in attention),
        )
        print(
            "Causal future attention max:",
            max(a.triu(diagonal=1).abs().max().item() for a in attention),
        )
    set_inspection(model, False)


if __name__ == "__main__":
    main()
