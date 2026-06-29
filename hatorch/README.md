# HATorch

## What is HATorch?

HATorch for _Hardware Aware PyTorch_ is a quantization-aware training framework built on top of PyTorch.

## WARNING

HATorch is still in the early stages of development. Important features are missing.

## Setup Guide

### Requirements

HATorch requires:

- Python ≥ 3.13  
- PyTorch (CPU or CUDA)  
- Torchvision  
- PytorchCV  
- Matplotlib  
- TQDM  

## Python Setup (pip / uv)

### CPU installation (pip)

```bash
pip install .[cpu, dev]
```

### CUDA 13.0 installation (pip)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install .[cu130, dev]
```

### CPU installation (uv)

```bash
uv sync --extra cpu --extra dev
```

### CUDA 13.0 installation (uv)

```bash
uv sync --extra cu130 --extra dev
```

## Getting Started

See `examples/vg11_cifar100.py`

## Experiences
