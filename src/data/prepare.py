import json
import os
from pathlib import Path

os.environ["RAYON_NUM_THREADS"] = "4"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from src.data.dataset import load_cosmopedia, load_fineweb

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TOKENIZER_PATH = Path("artifacts/tokenizer/bpe_32000.json")

VOCAB_SIZE = 32_000
EOT_TOKEN = "<|endoftext|>"

SEED = 42

FINEWEB_RATIO = 0.80
COSMOPEDIA_RATIO = 0.20

BATCH_DOCS = 16


OUTPUT_DIR = Path("artifacts/pretraining_data")
TRAIN_TARGET_TOKENS = 2_700_000_000
VAL_TARGET_TOKENS = 10_000_000
SHARD_SIZE = 100_000_000

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"

# uint16 storage only works if every token id fits in 16 bits.
assert VOCAB_SIZE <= 65_536, "VOCAB_SIZE too large for uint16 storage"


def write_atomic(array, path):
    """Write to a temp file, then rename, so a crash never leaves a
    truncated .bin behind."""
    tmp = path.with_name(path.name + ".tmp")
    array.tofile(tmp)
    os.replace(tmp, path)


def clean_output_dirs():
    """Remove leftovers from earlier runs."""
    for directory in (TRAIN_DIR, VAL_DIR):
        if directory.exists():
            for pattern in ("*.bin", "*.tmp"):
                for path in directory.glob(pattern):
                    path.unlink()
    if OUTPUT_DIR.exists():
        for name in ("metadata.json", "token_audit.json", "token_counts.npy"):
            (OUTPUT_DIR / name).unlink(missing_ok=True)


def next_documents(iterator, source_name, n):
    """Pull up to n non-empty documents."""
    texts = []
    while len(texts) < n:
        try:
            example = next(iterator)
        except StopIteration:
            if texts:
                break
            raise RuntimeError(
                f"Stream '{source_name}' was exhausted without collecting any documents!"
            )

        text = example.get("text", "")
        if isinstance(text, str) and text.strip():
            texts.append(text.replace(EOT_TOKEN, " "))
    return texts


def choose_source(fineweb_tokens, cosmopedia_tokens):
    """Pick whichever source is furthest behind its target ratio."""
    if fineweb_tokens / FINEWEB_RATIO <= cosmopedia_tokens / COSMOPEDIA_RATIO:
        return "fineweb"
    return "cosmopedia"


def document_stream(tokenizer, fineweb_iter, cosmopedia_iter, eot_id):
    """Yield (source, token_ids) one document at a time."""
    fineweb_tokens = 0
    cosmopedia_tokens = 0

    while True:
        source = choose_source(fineweb_tokens, cosmopedia_tokens)
        iterator = fineweb_iter if source == "fineweb" else cosmopedia_iter

        texts = next_documents(iterator, source, BATCH_DOCS)
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)

        for enc in encodings:
            ids = np.asarray(enc.ids + [eot_id], dtype=np.uint16)
            if source == "fineweb":
                fineweb_tokens += len(ids)
            else:
                cosmopedia_tokens += len(ids)
            yield source, ids


class ShardWriter:
    """Packs documents back to back into fixed-size shards. A document
    may cross a shard boundary. Counts token frequencies as it writes.
    offset → tracks how far we've progressed through the current document (ids).
    position → tracks how much space we've used in the current shard buffer.
    """

    def __init__(self, directory, shard_size, vocab_size):
        self.directory = directory
        self.shard_size = shard_size
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.position = 0
        self.num_shards = 0
        self.token_counts = np.zeros(vocab_size, dtype=np.int64)
        self.vocab_size = vocab_size

    def add(self, ids):
        offset = 0
        while offset < len(ids):
            n = min(self.shard_size - self.position, len(ids) - offset)
            self.buffer[self.position : self.position + n] = ids[offset : offset + n]
            self.position += n
            offset += n
            if self.position == self.shard_size:
                self._flush()

    def close(self):
        if self.position > 0:
            self._flush()

    def _flush(self):
        data = self.buffer[: self.position]
        self.token_counts += np.bincount(data, minlength=self.vocab_size)
        path = self.directory / f"train_{self.num_shards:02d}.bin"
        write_atomic(data, path)
        self.num_shards += 1
        self.position = 0


class SourceStats:
    def __init__(self):
        self.tokens = {"fineweb": 0, "cosmopedia": 0}

    def add(self, source, n):
        self.tokens[source] += n

    def summary(self):
        total = sum(self.tokens.values())
        return {
            "actual_tokens": total,
            "fineweb_tokens": self.tokens["fineweb"],
            "cosmopedia_tokens": self.tokens["cosmopedia"],
            "fineweb_fraction": self.tokens["fineweb"] / total,
            "cosmopedia_fraction": self.tokens["cosmopedia"] / total,
        }


def prepare_validation(tokenizer, fineweb_iter, cosmopedia_iter, eot_id):
    stream = document_stream(tokenizer, fineweb_iter, cosmopedia_iter, eot_id)
    stats = SourceStats()
    chunks = []
    total = 0

    with tqdm(total=VAL_TARGET_TOKENS, desc="Validation Tokens", unit="tok") as pbar:
        for source, ids in stream:
            chunks.append(ids)
            stats.add(source, len(ids))
            total += len(ids)
            pbar.update(len(ids))
            if total >= VAL_TARGET_TOKENS:
                break

    data = np.concatenate(chunks)
    del chunks
    write_atomic(data, VAL_DIR / "val.bin")
    return stats.summary()


def prepare_training(tokenizer, fineweb_iter, cosmopedia_iter, eot_id):
    stream = document_stream(tokenizer, fineweb_iter, cosmopedia_iter, eot_id)
    writer = ShardWriter(TRAIN_DIR, SHARD_SIZE, VOCAB_SIZE)
    stats = SourceStats()
    total = 0

    with tqdm(total=TRAIN_TARGET_TOKENS, desc="Training Tokens", unit="tok") as pbar:
        for source, ids in stream:
            writer.add(ids)
            stats.add(source, len(ids))
            total += len(ids)
            pbar.update(len(ids))

            if total >= TRAIN_TARGET_TOKENS:
                break

    writer.close()
    summary = stats.summary()
    summary["num_shards"] = writer.num_shards
    return summary, writer.token_counts


def save_token_audit(tokenizer, token_counts):
    np.save(OUTPUT_DIR / "token_counts.npy", token_counts)

    zero_ids = np.where(token_counts == 0)[0]
    rare_ids = np.where((token_counts >= 1) & (token_counts <= 9))[0]
    low_ids = np.where((token_counts >= 10) & (token_counts <= 99))[0]

    audit = {
        "zero_frequency_tokens": len(zero_ids),
        "frequency_1_to_9": len(rare_ids),
        "frequency_10_to_99": len(low_ids),
        "zero_frequency_details": [
            {"id": int(i), "token": tokenizer.id_to_token(int(i))} for i in zero_ids
        ],
        "frequency_1_to_9_details": [
            {
                "id": int(i),
                "token": tokenizer.id_to_token(int(i)),
                "count": int(token_counts[i]),
            }
            for i in rare_ids
        ],
    }

    with open(OUTPUT_DIR / "token_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=4, ensure_ascii=False)

    return audit


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    clean_output_dirs()

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != VOCAB_SIZE:
        raise ValueError(f"Expected vocab size {VOCAB_SIZE}, got {actual_vocab_size}")

    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    if eot_id is None:
        raise ValueError(f"EOT token '{EOT_TOKEN}' not found in tokenizer.")

    print(f"Loaded tokenizer from {TOKENIZER_PATH}")

    fineweb = iter(load_fineweb(seed=SEED))
    cosmopedia = iter(load_cosmopedia(seed=SEED + 1))

    print("\n>>> Stage 1: validation data")
    val_stats = prepare_validation(tokenizer, fineweb, cosmopedia, eot_id)
    print(f"Validation done: {val_stats['actual_tokens']:,} tokens")

    print("\n>>> Stage 2: training shards")
    train_stats, token_counts = prepare_training(tokenizer, fineweb, cosmopedia, eot_id)
    print(
        f"Training done: {train_stats['actual_tokens']:,} tokens "
        f"in {train_stats['num_shards']} shards"
    )

    print("\n>>> Stage 3: token frequency audit")
    audit = save_token_audit(tokenizer, token_counts)

    metadata = {
        "vocab_size": VOCAB_SIZE,
        "shard_size": SHARD_SIZE,
        "training": {"target_tokens": TRAIN_TARGET_TOKENS, **train_stats},
        "validation": {"target_tokens": VAL_TARGET_TOKENS, **val_stats},
        "token_audit": {
            "zero_frequency_tokens": audit["zero_frequency_tokens"],
            "frequency_1_to_9": audit["frequency_1_to_9"],
            "frequency_10_to_99": audit["frequency_10_to_99"],
        },
    }

    # Written last: its presence means the whole run finished.
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nPipeline complete.")
    print(json.dumps(metadata, indent=4))


if __name__ == "__main__":
    main()
