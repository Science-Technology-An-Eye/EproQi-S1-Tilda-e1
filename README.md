

---

# EproQi-S1-Tilda-e1

**EproQi-S1-Tilda-e1** is a lightweight, quantized Transformer model built on the **Tilda architecture**, optimized for **edge and low-latency inference** while remaining fully trainable at scale.

The model uses **BF16 / FP8 mixed precision**, rotary positional embeddings, and optimized attention kernels to achieve a strong balance between **performance, memory efficiency, and accuracy**, all within a **~70M parameter budget**.

---

## Model Overview

| Property            | Value                            |
| ------------------- | -------------------------------- |
| Architecture        | Tilda Transformer                |
| Parameters          | ~70M                             |
| Max Sequence Length | 2048                             |
| Attention           | MLA (Merged Linear Attention)    |
| Precision           | BF16 / FP8 (optional)            |
| Quantization        | Activation + weight quantization |
| Target              | Edge / low-VRAM inference        |

---

## Repository Structure

```
.
├── model.py        # Core Transformer model (Tilda architecture)
├── kernel.py       # Quantization & optimized GEMM kernels
├── train_i.py      # Training & fine-tuning pipeline
├── config.json     # Model & training configuration
├── checkpoints/    # Saved checkpoints
└── README.md
```

---

## model.py — Core Model Definition

`model.py` defines the complete **Transformer architecture**, including embeddings, attention, feed-forward blocks, and output head.

### Key Components

#### ModelArgs

A dataclass that centralizes all architecture hyperparameters:

* Embedding & hidden dimensions
* Attention head configuration
* Rotary embedding (RoPE) parameters
* Precision mode (`bf16` or `fp8`)
* KV / Q LoRA ranks
* Sequence length scaling options

This allows architecture changes **without modifying code**.

---

#### ParallelEmbedding

* Vocabulary-parallel embedding layer
* Supports distributed training via `torch.distributed`
* Reduces memory usage for large vocabularies

---

#### Linear / ColumnParallelLinear / RowParallelLinear

Custom linear layers with:

* BF16 or FP8 weights
* Block-wise quantization support
* Distributed parallelism
* Automatic fallback to standard `F.linear` when not quantized

These layers are the **foundation of quantized inference**.

---

#### RMSNorm

* Lightweight normalization
* Better numerical stability than LayerNorm
* Used before attention and MLP blocks

---

#### Rotary Positional Embeddings (RoPE)

Implemented via:

* `precompute_freqs_cis`
* `apply_rotary_emb`

Supports:

* Extended context lengths
* Frequency correction for long sequences
* Smooth extrapolation beyond original training length

---

#### MLA (Merged Linear Attention)

A highly optimized attention module featuring:

* Split QK heads (RoPE + non-RoPE)
* KV LoRA compression
* Two attention modes:

  * `naive` (explicit KV cache)
  * `absorb` (optimized KV + PE cache)
* Efficient causal masking
* Inference KV caching for autoregressive decoding

This design significantly reduces **memory bandwidth and latency**.

---

#### MLP

* SwiGLU-style feed-forward network
* Column/row parallel linear layers
* Optimized for throughput and quantization

---

#### Block

A standard Transformer block:

```
x → RMSNorm → Attention → Residual
  → RMSNorm → MLP → Residual
```

---

#### Transformer

The top-level model class:

* Embedding layer
* Stack of Transformer blocks
* Final RMSNorm
* Vocabulary-parallel output head
* Proper distributed logit gathering

Includes a **critical weight initialization fix** for:

* BF16 numerical stability
* FP8 quantization correctness
* Residual scaling by depth
* Output head logit scaling (×5.0) to avoid BF16 collapse

---

## kernel.py — Optimized Kernels

`kernel.py` contains low-level performance primitives used by `model.py`.

### Responsibilities

* Activation quantization (`act_quant`)
* Weight dequantization (`weight_dequant`)
* FP8 GEMM (`fp8_gemm`)
* Block-wise scaling for stable quantized math

These kernels allow:

* Lower VRAM usage
* Faster inference
* Minimal accuracy degradation

---

## train_i.py — Training Pipeline

`train_i.py` controls training and fine-tuning.

### Capabilities

* Binary packed dataset loading
* Gradient accumulation
* Learning rate warmup
* Gradient clipping
* Checkpoint save / resume
* Optional gradient checkpointing
* Distributed-ready design

---

### Training Command (Standard)

```bash
python train_i.py \
    --config config.json \
    --data-file ./preprocessed_data/train_packed.bin \
    --output-dir ./checkpoints \
    --batch-size 128 \
    --grad-accum 1 \
    --max-seq-len 2048 \
    --epochs 10 \
    --lr 6e-4 \
    --weight-decay 0.1 \
    --warmup-steps 2000 \
    --max-grad-norm 1.0
```

---

### Training with Gradient Checkpointing

```bash
python train_i.py \
  --config config.json \
  --data-file ./preprocessed_data/train_packed.bin \
  --output-dir ./checkpoints \
  --batch-size 128 \
  --grad-accum 1 \
  --max-seq-len 2048 \
  --epochs 10 \
  --lr 6e-4 \
  --weight-decay 0.1 \
  --warmup-steps 2000 \
  --max-grad-norm 1.0 \
  --gradient-checkpointing
```

This reduces memory usage at the cost of slightly slower training.

---

## config.json — Configuration File

The central configuration file controls:

* Model architecture parameters
* Precision mode (BF16 / FP8)
* Attention and RoPE settings
* Training defaults
* Dataset and checkpoint paths

**All experiments should be reproducible by editing this file only.**

---

## Checkpoint Management

List checkpoints:

```bash
ls -lh checkpoints/
```

Archive best model:

```bash
tar -czvf /workspace/best_model.tar.gz /workspace/checkpoints/best.pt
```

---

## GPU Cleanup (When Needed)

```bash
pkill -9 python
sleep 2
nvidia-smi --gpu-reset
nvidia-smi
```

Useful when recovering from OOM or stuck CUDA contexts.

---

## Design Philosophy

EproQi-S1-Tilda-e1 is built around:

1. **Quantization-first design**
2. **Edge and low-latency readiness**
3. **Numerical stability in BF16**
4. **Modular, readable code**
5. **Production-friendly training**

---

## License

Add your license information here.

---

If you want next, I can:

* 🔹 Add an **Inference README**
* 🔹 Write a **Hugging Face model card**
* 🔹 Create **architecture diagrams**
* 🔹 Review `train_i.py` for stability & speed
* 🔹 Estimate **exact parameter count from config**

You’re building this the *right way* — happy to help you polish it further 💪
