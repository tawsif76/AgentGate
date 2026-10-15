#!/usr/bin/env python3
"""
Plot a clean 16x16 grid of byte-level base symbols (00-FF),
highlighting zero-frequency base-byte tokens.

Inputs:
  - tokenizer JSON (ByteLevel BPE tokenizer)
  - token audit JSON

Default paths:
  ../artifacts/tokenizer/bpe_32000.json
  ../artifacts/pretraining_data/token_audit.json

Usage:
  python plot_zero_freq_base_bytes.py

  python plot_zero_freq_base_bytes.py \
      --tokenizer ../artifacts/tokenizer/bpe_32000.json \
      --audit ../artifacts/pretraining_data/token_audit.json \
      --out zero_frequency_base_bytes.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tokenizers import Tokenizer

# ---------------------------
# Styling
# ---------------------------
C_ZERO = "#c73e2b"  # highlighted zero-frequency base byte
C_OTHER = "#e6e6e6"  # other byte values
T_TITLE = "#1f2933"
T_MUTED = "#6b7783"


# ---------------------------
# ByteLevel mapping
# ---------------------------
def bytes_to_unicode():
    """
    Standard GPT-2 / Hugging Face ByteLevel reversible byte->Unicode mapping.
    Returns:
        dict[int, str]: byte value -> visible Unicode symbol
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


# ---------------------------
# Helpers
# ---------------------------
def is_base_symbol(token):
    """
    True if token is one of the 256 ByteLevel base symbols.
    """
    return len(token) == 1 and token in CHAR_TO_BYTE


def load_vocab(tokenizer_path):
    """
    Load tokenizer vocab from tokenizer JSON using Hugging Face tokenizers.
    Returns:
        vocab: dict[token_string -> token_id]
        id_to_token: dict[token_id -> token_string]
    """
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    id_to_token = {token_id: token for token, token_id in vocab.items()}
    return vocab, id_to_token


def load_zero_frequency_tokens(audit_path):
    """
    Load zero-frequency token entries from token_audit.json.
    Expected fields:
      - zero_frequency_tokens
      - zero_frequency_details: [{id, token}, ...]
    """
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    zero = audit["zero_frequency_details"]

    if len(zero) != audit["zero_frequency_tokens"]:
        raise ValueError(
            "zero_frequency_details length does not match zero_frequency_tokens"
        )

    return zero


def extract_zero_frequency_base_tokens(zero_entries, id_to_token):
    """
    From zero-frequency entries, keep only those that belong to
    the 256-symbol ByteLevel base alphabet, and map them to actual byte values.

    Returns:
        base_tokens: list of dicts with keys:
            - id
            - token
            - byte
        learned_tokens: list of non-base zero-frequency tokens
    """
    base_tokens = []
    learned_tokens = []

    for entry in zero_entries:
        token_id = entry["id"]
        token_from_audit = entry["token"]

        if token_id not in id_to_token:
            raise ValueError(f"Token ID {token_id} not found in tokenizer vocab")

        token_from_vocab = id_to_token[token_id]

        if token_from_vocab != token_from_audit:
            raise ValueError(
                f"Token mismatch for ID {token_id}: "
                f"audit={token_from_audit!r}, vocab={token_from_vocab!r}"
            )

        if is_base_symbol(token_from_vocab):
            base_tokens.append(
                {
                    "id": token_id,
                    "token": token_from_vocab,
                    "byte": CHAR_TO_BYTE[token_from_vocab],
                }
            )
        else:
            learned_tokens.append(
                {
                    "id": token_id,
                    "token": token_from_vocab,
                }
            )

    return base_tokens, learned_tokens


# ---------------------------
# Plotting
# ---------------------------
def plot_byte_grid(base_tokens, out_path):
    """
    Plot a minimal clean 16x16 grid.
    """
    status = np.zeros((16, 16), dtype=int)

    for t in base_tokens:
        b = t["byte"]
        status[b >> 4, b & 15] = 1

    fig, ax = plt.subplots(figsize=(7.8, 6.9), facecolor="white")

    ax.imshow(
        status,
        cmap=ListedColormap([C_OTHER, C_ZERO]),
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    # Put token symbol and hex byte value inside each highlighted cell
    for t in base_tokens:
        b = t["byte"]
        x = b & 15
        y = b >> 4

        ax.text(
            x,
            y - 0.14,
            t["token"],
            ha="center",
            va="center",
            fontsize=8.7,
            color="white",
            fontweight="bold",
        )
        ax.text(
            x,
            y + 0.19,
            f"{b:02X}",
            ha="center",
            va="center",
            fontsize=7.2,
            color="white",
        )

    # Ticks and labels
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"{i:X}" for i in range(16)], fontsize=10, color=T_MUTED)

    ax.set_yticks(range(16))
    ax.set_yticklabels([f"{i:X}" for i in range(16)], fontsize=10, color=T_MUTED)

    # Gridlines
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)

    ax.tick_params(which="both", length=0)

    # Axis labels
    ax.set_xlabel("Low hexadecimal digit", fontsize=11, color=T_MUTED, labelpad=8)
    ax.set_ylabel("High hexadecimal digit", fontsize=11, color=T_MUTED, labelpad=8)

    # Simple title only
    ax.set_title("Zero-Frequency Base Bytes", fontsize=14, color=T_TITLE, pad=12)

    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.pdf')}")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot a clean 16x16 grid of zero-frequency byte-level base symbols."
    )
    parser.add_argument(
        "--tokenizer",
        default="../artifacts/tokenizer/bpe_32000.json",
        help="Path to tokenizer JSON",
    )
    parser.add_argument(
        "--audit",
        default="../artifacts/pretraining_data/token_audit.json",
        help="Path to token audit JSON",
    )
    parser.add_argument(
        "--out",
        default="zero_frequency_base_bytes.png",
        help="Output figure path",
    )

    args = parser.parse_args()

    vocab, id_to_token = load_vocab(args.tokenizer)
    zero_entries = load_zero_frequency_tokens(args.audit)
    base_tokens, learned_tokens = extract_zero_frequency_base_tokens(
        zero_entries, id_to_token
    )

    # Optional sanity checks
    print(f"Total zero-frequency tokens: {len(zero_entries)}")
    print(f"Zero-frequency base-byte tokens: {len(base_tokens)}")
    print(f"Zero-frequency learned tokens: {len(learned_tokens)}")

    if learned_tokens:
        print("Zero-frequency learned token(s):")
        for t in learned_tokens:
            print(f"  id {t['id']}: {t['token']!r}")

    plot_byte_grid(base_tokens, args.out)


if __name__ == "__main__":
    main()
