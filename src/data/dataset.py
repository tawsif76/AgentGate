from datasets import interleave_datasets, load_dataset

DATASET_NAME = "HuggingFaceTB/smollm-corpus"

SHUFFLE_BUFFER_SIZE = 10000


def _add_fineweb_source(example):
    return {"source": "fineweb"}


def _add_cosmopedia_source(example):
    return {"source": "cosmopedia"}


def load_fineweb(
    seed=42,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
):
    dataset = load_dataset(
        DATASET_NAME,
        "fineweb-edu-dedup",
        split="train",
        streaming=True,
    )

    # We only need the text itself.
    dataset = dataset.select_columns(["text"])

    # Add source information so later we can measure
    # FineWeb and Cosmopedia separately.
    dataset = dataset.map(_add_fineweb_source)

    # Approximate streaming shuffle.
    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size,
    )

    return dataset


def load_cosmopedia(
    seed=42,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
):
    dataset = load_dataset(
        DATASET_NAME,
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )

    dataset = dataset.select_columns(["text"])

    dataset = dataset.map(_add_cosmopedia_source)

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size,
    )

    return dataset


def load_training_dataset(
    fineweb_prob=0.8,
    cosmopedia_prob=0.2,
    seed=42,
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
):
    fineweb = load_fineweb(
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    cosmopedia = load_cosmopedia(
        seed=seed + 1,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    dataset = interleave_datasets(
        [
            fineweb,
            cosmopedia,
        ],
        probabilities=[
            fineweb_prob,
            cosmopedia_prob,
        ],
        seed=seed,
    )

    return dataset


if __name__ == "__main__":
    dataset = load_training_dataset()

    for i, example in enumerate(dataset):
        print("=" * 80)
        print("Source:", example["source"])
        print()
        print(example["text"][:500])

        if i == 4:
            break
