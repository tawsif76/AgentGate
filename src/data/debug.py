import numpy as np
from tokenizers import Tokenizer

from src.data.dataset import load_cosmopedia, load_fineweb

SEED = 42
EOT_TOKEN = "<|endoftext|>"


def tokenize_document(tokenizer, text, eot_id):
    clean_text = text.replace(EOT_TOKEN, " ")
    ids = tokenizer.encode(clean_text, add_special_tokens=False).ids
    ids.append(eot_id)
    print("\ninside:", ids)
    return np.asarray(ids, dtype=np.uint16)


def choose_source(fineweb_tokens, cosmopedia_tokens):
    if fineweb_tokens / 0.80 <= cosmopedia_tokens / 0.20:
        return "fineweb"
    return "cosmopedia"


def next_documents(iterator, source_name, n):
    """Pull up to n non-empty documents. Text is kept unchanged, except
    that any literal EOT string is removed: the tokenizer would
    otherwise encode it as the real EOT token in the middle of a document."""
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


def document_stream(tokenizer, fineweb_iter, cosmopedia_iter, eot_id):
    fineweb_tokens = 0
    cosmopedia_tokens = 0

    while True:
        source = choose_source(fineweb_tokens, cosmopedia_tokens)
        iterator = fineweb_iter if source == "fineweb" else cosmopedia_iter

        texts = next_documents(iterator, source, n=4)
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)

        for enc in encodings:
            ids = np.asanyarray(enc.ids + [eot_id], dtype=np.uint16)
            if source == "fineweb":
                fineweb_tokens += len(ids)
            else:
                cosmopedia_tokens += len(ids)
            yield source, ids


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("artifacts/tokenizer/bpe_32000.json")
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    print(eot_id)
    text = "Hello, world! This is a test document."
    token_ids = tokenize_document(tokenizer, text, eot_id)
    print("\nToken IDs:", token_ids)
    fineweb = iter(load_fineweb(seed=SEED))
    cosmopedia = iter(load_cosmopedia(seed=SEED + 1))
    document_stream(tokenizer, fineweb, cosmopedia, eot_id)
