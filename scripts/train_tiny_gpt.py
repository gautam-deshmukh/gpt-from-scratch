import json
import math
import pathlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = pathlib.Path("data/raw")
TOKENIZER_PATH = pathlib.Path("data/processed/char_tokenizer.json")
CHECKPOINT_DIR = pathlib.Path("checkpoints")


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
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--checkpoint_name", type=str, default="tiny_gpt_checkpoint.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    text = load_corpus_text()
    tokenizer = CharTokenizer.load(TOKENIZER_PATH)

    dataset = CharDataset(text, tokenizer, seq_len=args.block_size, stride=1)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint_name

    model.train()
    step = 0
    final_loss = None

    while step < args.max_steps:
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

            if step % args.print_every == 0:
                print(f"step {step:04d} | loss {final_loss:.4f}")

            step += 1
            if step >= args.max_steps:
                break

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": final_loss,
        "vocab_size": tokenizer.vocab_size,
        "block_size": args.block_size,
        "n_embd": args.n_embd,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
