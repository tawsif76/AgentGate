import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.data.dataset import load_training_dataset

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCAB_SIZES = [
    24_000,
    32_000,
]

EOT_TOKEN = "<|endoftext|>"

TOKENIZER_DIR = Path("artifacts/tokenizer")

SEED = 42

TRAIN_DOCUMENTS = 100_000
EVAL_DOCUMENTS = 20_000

# 20K eval docs -> 10 blocks
EVAL_BLOCK_SIZE = 2_000


# --------------------------------------------------
# Tokenizer creation
# --------------------------------------------------


def create_tokenizer():

    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer


def create_trainer(vocab_size):

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            EOT_TOKEN,
        ],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    return trainer


# --------------------------------------------------
# Dataset iterator
# --------------------------------------------------


def text_iterator(
    dataset,
    start_document,
    num_documents,
):
    """
    Yield exactly num_documents non-empty documents.

    Example:

    start_document = 0
    num_documents = 100_000

    -> first 100K valid documents

    start_document = 100_000
    num_documents = 20_000

    -> next 20K valid documents
    """

    valid_index = 0
    yielded = 0

    for example in dataset:
        text = example["text"]

        if not text or not text.strip():
            continue

        if valid_index < start_document:
            valid_index += 1
            continue

        if yielded >= num_documents:
            break

        yield text

        valid_index += 1
        yielded += 1


# --------------------------------------------------
# Train tokenizer
# --------------------------------------------------


def train_tokenizer(vocab_size):

    print()
    print("=" * 70)
    print(f"Training tokenizer | Vocab = {vocab_size:,}")
    print(f"Training documents = {TRAIN_DOCUMENTS:,}")
    print("=" * 70)

    # Same seed means both tokenizers
    # see the same training stream.
    dataset = load_training_dataset(seed=SEED)

    tokenizer = create_tokenizer()

    trainer = create_trainer(vocab_size)

    tokenizer.train_from_iterator(
        text_iterator(
            dataset=dataset,
            start_document=0,
            num_documents=TRAIN_DOCUMENTS,
        ),
        trainer=trainer,
        length=TRAIN_DOCUMENTS,
    )

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = TOKENIZER_DIR / f"bpe_{vocab_size}.json"

    tokenizer.save(str(path))

    print()
    print(f"Saved tokenizer: {path}")

    print(
        "Actual vocab size:",
        tokenizer.get_vocab_size(),
    )


def evaluate_tokenizer(vocab_size):

    path = TOKENIZER_DIR / f"bpe_{vocab_size}.json"

    tokenizer = Tokenizer.from_file(str(path))

    # Recreate exactly the same shuffled stream.
    dataset = load_training_dataset(seed=SEED)

    eval_texts = text_iterator(
        dataset=dataset,
        start_document=TRAIN_DOCUMENTS,
        num_documents=EVAL_DOCUMENTS,
    )

    token_counts = Counter()

    total_tokens = 0
    total_characters = 0
    total_documents = 0

    # Used to see whether compression
    # is stable across different chunks.
    block_chars = 0
    block_tokens = 0
    block_documents = 0

    block_scores = []

    for text in eval_texts:
        encoding = tokenizer.encode(text)

        ids = encoding.ids

        # Overall statistics
        token_counts.update(ids)

        total_tokens += len(ids)
        total_characters += len(text)
        total_documents += 1

        # Current evaluation block
        block_tokens += len(ids)
        block_chars += len(text)
        block_documents += 1

        # Finish one 2K-document block
        if block_documents == EVAL_BLOCK_SIZE:
            block_chars_per_token = block_chars / block_tokens

            block_scores.append(block_chars_per_token)

            print(
                f"Vocab {vocab_size:,}: "
                f"evaluated "
                f"{total_documents:,} / "
                f"{EVAL_DOCUMENTS:,} docs"
            )

            block_chars = 0
            block_tokens = 0
            block_documents = 0

    # In case the final block is incomplete
    if block_documents > 0:
        block_scores.append(block_chars / block_tokens)

    # --------------------------------------------------
    # Compression
    # --------------------------------------------------

    characters_per_token = total_characters / total_tokens

    # --------------------------------------------------
    # Vocabulary tail
    # --------------------------------------------------

    actual_vocab_size = tokenizer.get_vocab_size()

    eot_id = tokenizer.token_to_id(EOT_TOKEN)

    normal_token_ids = [
        token_id for token_id in range(actual_vocab_size) if token_id != eot_id
    ]

    normal_vocab_size = len(normal_token_ids)

    unseen_tokens = sum(token_counts[token_id] == 0 for token_id in normal_token_ids)

    rare_tokens_1_to_9 = sum(
        0 < token_counts[token_id] < 10 for token_id in normal_token_ids
    )

    unseen_percent = unseen_tokens / normal_vocab_size * 100

    rare_percent = rare_tokens_1_to_9 / normal_vocab_size * 100

    # --------------------------------------------------
    # Stability
    # --------------------------------------------------

    block_mean = mean(block_scores)

    block_std = stdev(block_scores) if len(block_scores) > 1 else 0.0

    return {
        # Metadata
        "vocab_size": actual_vocab_size,
        "documents": total_documents,
        "tokens": total_tokens,
        # Actual decision metrics
        "characters_per_token": characters_per_token,
        "unseen_percent": unseen_percent,
        "rare_percent_1_to_9": rare_percent,
        "block_mean": block_mean,
        "block_std": block_std,
    }


def save_results(results):

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = TOKENIZER_DIR / "comparison_results.json"

    output = {
        "train_documents": TRAIN_DOCUMENTS,
        "eval_documents": EVAL_DOCUMENTS,
        "eval_block_size": EVAL_BLOCK_SIZE,
        "seed": SEED,
        "results": results,
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=4,
        )

    print()
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    for vocab_size in VOCAB_SIZES:
        train_tokenizer(vocab_size=vocab_size)

    results = []

    for vocab_size in VOCAB_SIZES:
        result = evaluate_tokenizer(vocab_size=vocab_size)

        results.append(result)

    save_results(results)
