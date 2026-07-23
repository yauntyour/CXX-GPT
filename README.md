# CXX-GPT

A from-scratch implementation of a **Generative Pre-trained Transformer (GPT)** language model in **C++17 with CUDA acceleration**. Built as an educational project to understand transformer internals through hands-on implementation — no Python ML frameworks, no automatic differentiation, just C++, CUDA, and manual gradient computation.

## Features

- **Full GPT Model** — Decoder-only transformer with multi-head causal self-attention, GELU activation, pre-norm LayerNorm, and embedding layers
- **Manual Forward/Backward** — Every layer implements its own forward and backward pass; gradients flow explicitly through the computation graph
- **Custom Tensor Library (TensorN)** — N-dimensional tensor operations on CPU (OpenBLAS) and GPU (CUDA/cuBLAS)
- **CUDA Kernels** — Fused kernels for attention masking, layer norm, cross-entropy loss, embedding lookup, AdamW optimizer step, and more
- **BPE Tokenizer** — Full Byte-Pair Encoding tokenizer implemented in C++ with UTF-8 and CJK support
- **Training Pipeline** — AdamW optimizer, checkpoint save/load with interrupt (Ctrl+C) resume, streaming JSONL dataset, and built-in numerical gradient checking
- **Interactive Demo** — Chat with the trained model using top-k sampling

## Quick Start

### Prerequisites

- CMake 3.18+
- Visual Studio 2022 (MSVC)
- CUDA 12+ (optional but recommended)
- Python 3 (for dataset pre-encoding)

### Build

```bash
cmake -B build -G "Visual Studio 17 2022" -DTENSORN_ENABLE_CUDA=ON
cmake --build build --config Release
```

Binaries are output to `build/bin/`.

### Train

```bash
./build/bin/Release/gpt_train.exe
```

Trains an ~85M parameter GPT model on the provided pretraining dataset, saving checkpoints to `checkpoint.bin` and the final model to `model.bin`.

### Chat

```bash
./build/bin/Release/gpt_demo.exe
```

Loads a trained model and runs an interactive chat session.

## Project Structure

```
CXX-GPT/
├── CMakeLists.txt              # Root build configuration
├── src/
│   ├── gpt.hpp                 # Core model: GPT, LayerNorm, Linear, CrossEntropyLoss, AdamW, BPE tokenizer
│   ├── dataset.hpp             # Streaming JSONL dataset loader
│   ├── train.cpp               # Training loop
│   ├── demo.cpp                # Interactive chat demo
│   ├── cuda_kernels.cuh        # CUDA kernel declarations
│   └── cuda_kernels.cu         # CUDA kernel implementations
├── tokenizer/
│   ├── tokenizer.json          # BPE tokenizer model
│   └── tokenizer_config.json   # Tokenizer configuration
├── dataset/                    # Training / validation / test data (JSONL + binary)
├── scripts/
│   └── preencode.py            # Python dataset pre-encoding script
├── TensorN/                    # Custom tensor library (subproject)
└── build/                      # Build artifacts
```

## Architecture

| Component | Description |
|---|---|
| **GPT** | Decoder-only transformer with token + position embeddings, N transformer blocks, and lm_head |
| **Block** | Pre-norm LayerNorm → Causal Self-Attention → LayerNorm → MLP (GELU) |
| **TensorN** | Header-only C++ tensor library with CPU (OpenBLAS) and GPU (CUDA) backends |
| **CUDA Kernels** | Fused forward/backward kernels for attention, loss, optimizers, and normalization |
| **Tokenizer** | BPE tokenizer with 4096 vocab, UTF-8 support, and special token handling |
| **Dataset** | Streaming JSONL reader with binary pre-encoding for fast loading |

## Model Configuration

Default config (~85M parameters):
- `vocab_size`: 4096
- `block_size`: 64
- `n_embd`: 384
- `n_layer`: 16

## License

MIT License — see [LICENSE](LICENSE) for details.
