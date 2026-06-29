# Quantization Aware Training of Conv-TasNet speech separation model

Conv-TasNet is a fully-convolutional single-channel time-domain speech separation network. This work intend to evaluate Quantization Aware Training (QAT) possibilities on this model using HATorch quantization framework.

> Y. Luo and N. Mesgarani, “Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, no. 8, pp. 1256–1266, Aug. 2019, doi: 10.1109/TASLP.2019.2915167.
>
> J. Cosentino, M. Pariente, S. Cornell, A. Deleforge, and E. Vincent, “LibriMix: An Open-Source Dataset for Generalizable Speech Separation.” Accessed: Mar. 25, 2026. [Online]. Available: http://arxiv.org/abs/2005.11262
>
> B. Barbe and R. Bouarah, “HATorch.” [Online]. Available: https://gitlab.inria.fr/romain-bouarah-phd/hatorch

# Usage

Create a virtual environment
```bash
uv venv
uv sync
```

Get LibriMix dataset: https://github.com/JorisCos/LibriMix

Configure hyperparameters and environment in [conf.yml](./conf.yml)

## Training Conv-TasNet model

```python
python src/train.py
```

## Quantization Aware Training

```python
python src/quantize.py
```

## Evaluation

```python
python src/evaluate.py
```