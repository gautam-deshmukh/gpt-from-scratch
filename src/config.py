from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 16
    block_size: int = 128
    max_iters: int = 2000
    eval_interval: int = 200
    learning_rate: float = 3e-4

    vocab_size: int = 16000
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True

    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
