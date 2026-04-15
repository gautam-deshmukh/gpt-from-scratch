# GPT from Scratch

A small, educational decoder-only GPT project built from scratch in Python and PyTorch. This repo walks through the full language-model pipeline on a compact curated corpus: raw text collection, character-level tokenization, fixed-length sequence dataset creation, a tiny GPT-style transformer, training, checkpoint saving, and autoregressive text generation.

## Why I built this

I wanted to understand the core mechanics of language modeling at a lower level instead of relying entirely on high-level libraries. The goal of this project is not to build a production LLM, but to implement and verify the core pieces of a GPT-style model end to end.

## What this project does

This project currently includes:

- Corpus collection in plain UTF-8 text files under `data/raw/`
- A reusable character-level tokenizer saved to `data/processed/char_tokenizer.json`
- A PyTorch dataset and dataloader for next-token prediction
- A tiny decoder-only GPT-style model with:
  - token embeddings
  - positional embeddings
  - one causal self-attention layer
  - layer normalization
  - linear output projection to vocabulary logits
- A training script using AdamW and checkpoint saving
- A generation script for autoregressive text sampling from a saved checkpoint

## Project structure

```text
gpt-from-scratch/
├── data/
│   ├── raw/
│   │   ├── abstracts/
│   │   ├── docs/
│   │   ├── notes/
│   │   └── tutorials/
│   └── processed/
│       └── char_tokenizer.json
├── scripts/
│   ├── inspect_corpus.py
│   ├── build_char_tokenizer.py
│   ├── build_dataset.py
│   ├── tiny_gpt.py
│   ├── train_tiny_gpt.py
│   └── generate_text.py
├── checkpoints/
│   └── tiny_gpt_checkpoint.pt
├── results/
│   ├── training_summary.txt
│   └── sample_generations.txt
└── README.md
```

## Corpus

The training corpus is intentionally small and high-signal. It consists of cleaned ML-related text, including tokenizer tutorials, dataset tutorials, and short paraphrased research abstracts. The purpose of using a small corpus is to make debugging and inspection easy during early experiments.

## Model overview

The current model is a minimal GPT-style decoder with a single causal self-attention layer.

### Pipeline

1. Load all text files from `data/raw/`
2. Build or load a character-level tokenizer
3. Encode the corpus into token IDs
4. Slice the token stream into fixed-length `(x, y)` training pairs
5. Train the model with next-token cross-entropy loss
6. Save a checkpoint
7. Generate text autoregressively from a prompt

### Current training configuration

- Tokenization: character-level
- Vocabulary size: 82
- Context length: 32
- Batch size: 16
- Embedding dimension: 64
- Optimizer: AdamW
- Training steps: 300
- Device used: CPU

## Results

The model trained successfully and showed a meaningful reduction in loss over 300 steps.

- Initial loss: about 4.6381
- Later loss range: about 2.62 to 2.78
- Checkpoint saved to `checkpoints/tiny_gpt_checkpoint.pt`

Because the model is very small and the corpus is tiny, generated text is noisy and only partially coherent. That is expected. The point of this project is to demonstrate a working end-to-end implementation of a GPT-style training and generation pipeline.

### Example generation

Prompt:

```text
Attention
```

Sample output:

```text
Attention t atokokeseke onis)or fritenizes ierendem nented thereco. -r.l yh okingery we aokheitetiegininizening cheue al te to an zas toizeduwas oh sus tokpers intinsetas to intang_ins ietok iindetous itavs th
```

## How to run

### 1. Inspect the corpus

```bash
python3 scripts/inspect_corpus.py
```

### 2. Build the tokenizer

```bash
python3 scripts/build_char_tokenizer.py
```

### 3. Build and inspect the dataset

```bash
python3 scripts/build_dataset.py
```

### 4. Run the tiny GPT forward pass

```bash
python3 scripts/tiny_gpt.py
```

### 5. Train the model

```bash
python3 scripts/train_tiny_gpt.py
```

### 6. Generate text from the trained checkpoint

```bash
python3 scripts/generate_text.py
```

## What I learned

This project helped me understand:

- how tokenization choices affect the modeling pipeline
- how next-token datasets are constructed from raw text
- how causal self-attention works in a decoder-only transformer
- how logits and cross-entropy loss are used for language modeling
- how to save, reload, and sample from trained PyTorch checkpoints

## Current limitations

- Character-level tokenization is simple but inefficient
- The corpus is extremely small
- The model has only one attention layer
- Training was brief and done on CPU
- Generated text is only weakly coherent

## Next steps

- Add a BPE tokenizer implementation and compare it with character-level tokenization
- Run longer training or try a slightly larger model
- Save multiple generations with different prompts and temperatures
- Compare context lengths or embedding sizes
- Improve experiment tracking and reporting

## Notes

This is an educational, from-scratch implementation project focused on learning the fundamentals of language modeling and transformer training.
