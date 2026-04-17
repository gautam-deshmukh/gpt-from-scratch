# GPT From Scratch

A small from-scratch implementation of a decoder-only GPT-style language model and tokenizer in PyTorch.

This project includes:
- corpus preparation from local text files,
- character-level tokenization,
- a custom dataset class for next-token prediction,
- training for a tiny GPT-style model,
- text generation from a saved checkpoint,
- and a small hyperparameter comparison experiment.

## Project Overview

The goal of this project was to build and understand the core pieces of a GPT-style language model pipeline from scratch rather than relying on high-level training libraries.

The repository currently includes:
- tokenizer creation and vocabulary serialization,
- dataset preparation for autoregressive language modeling,
- model training with causal self-attention,
- checkpoint saving,
- sample text generation,
- and a comparison of embedding sizes.

## Repository Structure

```text
gpt-from-scratch/
├── checkpoints/
├── data/
│   ├── processed/
│   └── raw/
├── results/
├── scripts/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/gautam-deshmukh/gpt-from-scratch.git
cd gpt-from-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Pipeline

### 1. Build the tokenizer

```bash
python3 scripts/build_tokenizer.py
```

### 2. Inspect the dataset logic

```bash
python3 scripts/build_dataset.py
```

### 3. Train the model

```bash
python3 scripts/train_tiny_gpt.py
```

## Text Generation

To generate text from a saved checkpoint:

```bash
python3 scripts/generate.py
```

Generated samples are also stored in the `results/` directory.

## Experiment

This repository includes a small controlled experiment comparing embedding dimension sizes:

- `n_embd = 64`
- `n_embd = 128`

The summary and observations are recorded in:

```text
results/embedding_comparison.txt
```

In this short training setup, the 64-dimensional version slightly outperformed the 128-dimensional version on final observed loss, showing that a larger embedding size did not automatically improve performance on a tiny dataset and short training budget.

## Notes

This is a small educational implementation intended to understand the mechanics of tokenizer construction, dataset preparation, autoregressive training, and GPT-style text generation at a low level.

## Future Improvements

- add multi-head attention,
- support train/validation splits,
- improve checkpoint naming and experiment tracking,
- add richer text generation controls,
- and expand beyond character-level tokenization.
