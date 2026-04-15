import json
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader

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
        chars = list(dict.fromkeys(chars))
        if self.UNK not in chars:
            chars.append(self.UNK)
        self.chars = chars
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def load(cls, path: pathlib.Path):
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["chars"], unk_token=obj.get("unk_token", "<UNK>"))

    def encode(self, text: str):
        unk_id = self.char_to_id[self.UNK]
        return [self.char_to_id.get(ch, unk_id) for ch in text]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, self.UNK) for i in ids)


class CharDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=32, stride=1):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.data = tokenizer.encode(text)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.data[start + 1:end + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def main():
    text = load_corpus_text()
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)

    dataset = CharDataset(text, tokenizer, seq_len=32, stride=1)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"Corpus characters: {len(text)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Dataset examples: {len(dataset)}")

    x_batch, y_batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print("x_batch:", x_batch.shape)
    print("y_batch:", y_batch.shape)

    print("\nFirst sample from batch, decoded:")
    print("x:", tokenizer.decode(x_batch[0].tolist()))
    print("y:", tokenizer.decode(y_batch[0].tolist()))
    

if __name__ == "__main__":
    main()

