# GPT From Scratch

A personal project where I build a GPT-style language model and tokenizer from scratch in PyTorch, heavily inspired by Andrej Karpathy’s educational code.

## Overview

The transformer model in this repository started from Karpathy’s minGPT-style implementation of a decoder-only GPT architecture. I am using that code as a learning scaffold and then extending it with my own additions: a custom tokenizer pipeline, configuration setup, training loop, sampling utilities, and experiment scripts.

The goal is to turn a minimal reference implementation into a complete, well-structured project that I fully understand and can modify, rather than treating large language models as a black box.

Concretely, this project aims to:
- Train a byte-level BPE tokenizer on custom text data
- Use that tokenizer with a decoder-only transformer model in PyTorch
- Implement training, sampling, and evaluation scripts
- Run experiments with model size, context length, and generation settings
- Document what I learn along the way
