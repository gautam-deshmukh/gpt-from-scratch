from pathlib import Path

import torch
from tokenizers import Tokenizer


RAW_DIR = Path("data/raw")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
OUT_PATH = Path("data/processed/dataset.pt")


def load_text_files(raw_dir):
    text_parts = []
    files = sorted(raw_dir.rglob("*.txt"))
    if not files:
        raise FileNotFoundError("No .txt files found in data/raw")
    for path in files:
        text_parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(text_parts)


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Tokenizer not found at tokenizer/tokenizer.json. "
            "Run tokenizer/train_tokenizer.py first."
        )

    text = load_text_files(RAW_DIR)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    encoded = tokenizer.encode(text)
    ids = torch.tensor(encoded.ids, dtype=torch.long)

    n = int(0.9 * len(ids))
    train_ids = ids[:n]
    val_ids = ids[n:]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train": train_ids,
            "val": val_ids,
            "vocab_size": tokenizer.get_vocab_size(),
        },
        OUT_PATH,
    )

    print(f"saved dataset to {OUT_PATH}")
    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens: {len(val_ids):,}")
    print(f"vocab size: {tokenizer.get_vocab_size():,}")


if __name__ == "__main__":
    main()
