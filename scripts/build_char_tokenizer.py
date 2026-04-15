import json
import pathlib
from collections import Counter

DATA_ROOT = pathlib.Path("data/raw")
TOKENIZER_PATH = pathlib.Path("data/processed/char_tokenizer.json")


def iter_text(root: pathlib.Path):
    for path in root.rglob("*.txt"):
        with path.open("r", encoding="utf-8") as f:
            yield f.read()


def load_corpus_text():
    return "\n".join(iter_text(DATA_ROOT))


class CharTokenizer:
    def __init__(self, chars, unk_token="<UNK>"):
        self.UNK = unk_token
        # ensure UNK is last
        chars = list(dict.fromkeys(chars))  # dedupe, keep order
        if self.UNK not in chars:
            chars.append(self.UNK)
        self.chars = chars
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_corpus(cls, text: str, unk_token="<UNK>"):
        chars = sorted(set(text))
        return cls(chars, unk_token=unk_token)

    def encode(self, text: str):
        unk_id = self.char_to_id[self.UNK]
        return [self.char_to_id.get(ch, unk_id) for ch in text]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, self.UNK) for i in ids)

    def save(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "unk_token": self.UNK,
            "chars": self.chars,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)  # standard JSON save pattern [web:478][web:481]

    @classmethod
    def load(cls, path: pathlib.Path):
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["chars"], unk_token=obj.get("unk_token", "<UNK>"))

def main():
    corpus = load_corpus_text()
    print(f"Loaded corpus with {len(corpus)} characters")

    # build tokenizer from corpus
    tokenizer = CharTokenizer.from_corpus(corpus)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # save to disk
    tokenizer.save(TOKENIZER_PATH)
    print(f"Saved tokenizer to {TOKENIZER_PATH}")

    # reload and quick round-trip test
    loaded = CharTokenizer.load(TOKENIZER_PATH)
    sample = "attention is all you need"
    ids = loaded.encode(sample)
    back = loaded.decode(ids)
    print("Sample:", sample)
    print("Decoded after reload:", back)


if __name__ == "__main__":
    main()
