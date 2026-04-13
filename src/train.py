import os
from pathlib import Path

import torch

from src.config import TrainConfig
from src.model import GPT, GPTConfig


def load_data(path):
    data = torch.load(path)
    return data["train"], data["val"]


def get_batch(split_data, batch_size, block_size, device):
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device, eval_iters=50):
    out = {}
    model.eval()
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_data, config.batch_size, config.block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def main():
    train_config = TrainConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    data_path = Path("data/processed/dataset.pt")
    if not data_path.exists():
        raise FileNotFoundError(
            "Expected tokenized dataset at data/processed/dataset.pt. "
            "Create this in the preprocessing step before training."
        )

    train_data, val_data = load_data(data_path)

    model_config = GPTConfig(
        block_size=train_config.block_size,
        vocab_size=train_config.vocab_size,
        n_layer=train_config.n_layer,
        n_head=train_config.n_head,
        n_embd=train_config.n_embd,
        dropout=train_config.dropout,
        bias=train_config.bias,
    )

    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        device_type=device,
    )

    os.makedirs("outputs/checkpoints", exist_ok=True)

    for step in range(train_config.max_iters):
        if step % train_config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, train_config, device)
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data, train_config.batch_size, train_config.block_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint_path = Path("outputs/checkpoints/gpt_checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_config": train_config.__dict__,
            "model_config": model_config.__dict__,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
