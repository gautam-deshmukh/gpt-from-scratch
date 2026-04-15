import json
import math
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        return out


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attention = CausalSelfAttention(n_embd, block_size)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.attention(x)
        x = self.ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss


def main():
    text = load_corpus_text()
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)

    block_size = 32
    batch_size = 8

    dataset = CharDataset(text, tokenizer, seq_len=block_size, stride=1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_batch, y_batch = next(iter(dataloader))

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=64,
    )

    logits, loss = model(x_batch, y_batch)

    print(f"x_batch shape: {x_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()

