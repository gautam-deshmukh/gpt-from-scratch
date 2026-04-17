import json
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

TOKENIZER_PATH = pathlib.Path("data/processed/char_tokenizer.json")
CHECKPOINT_PATH = pathlib.Path("checkpoints/tiny_gpt_checkpoint.pt")


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
        return att @ v


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attention = CausalSelfAttention(n_embd, block_size)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.attention(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CharTokenizer.load(TOKENIZER_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    block_size = checkpoint["block_size"]
    n_embd = checkpoint["n_embd"]

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = "The"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200)

    text = tokenizer.decode(output_ids[0].tolist())

    print("\nGenerated text:\n")
    print(text)


if __name__ == "__main__":
    main()
