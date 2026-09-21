from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "cosmopedia-v2",
    split="train",
    streaming=True,
)

example = next(iter(dataset))
print("Available keys:", list(example.keys()))
print("\nFirst 500 characters of text:\n")
print(example["text"][:2000])
