# MLX_Train: Native MultiLoRA-MoE Training & Serving Backend for Apple Silicon

**MLX_Train** (`mati_moe`) is a high-performance training and dynamic serving backend built on **Apple MLX**, designed to train disjoint domain specialist LoRA adapters (`Theory`, `Agentic`, `ASM/Systems`) and combine them into a unified **MultiLoRA Mixture-of-Experts (MoE)** stack on Apple Silicon.

---

## 🏛️ Architecture Overview

Instead of destructive linear weight merging (MergeKit), `MLX_Train` trains separate low-rank specialist adapters over a shared base model (`Gemma 4 12B / 26B`) and dynamically routes tokens at inference time via **Residual Boosting (MILE)** across layers 8–47:

```
                          ┌──► Theory Expert (Math / Security / Cyber)
                          │
Gemma 4 Base Model ───────┼──► Agentic Expert (Tool Calling / Structured JSON)
                          │
                          └──► ASM / Systems Expert (Decompilation / Auditing)
                                    │
                                    ▼
                      Dynamic Top-K Sparse Gating Router
```

---

## 💻 Hardware Requirements (Apple Silicon)

`MLX_Train` leverages Apple Silicon's unified memory architecture and Metal Execution Engine for zero-latency tensor allocation.

| Requirement | Minimum Specification | Recommended Production Setup |
| :--- | :--- | :--- |
| **System SoC** | Apple M1 / M2 / M3 / M4 / M5 Max or Ultra | **Apple M5 Max / M4 Max / M3 Max** |
| **Unified Memory (RAM)** | **64 GB** (up to `4096` sequence length) | **128 GB** (up to `8192` sequence length) |
| **Operating System** | macOS 14.0+ (Sonoma or Sequoia) | macOS 15.0+ |
| **MLX Version** | `mlx >= 0.22.0`, `mlx-lm >= 0.21.0` | Latest release |

---

## ⚡ Verified Benchmark Speeds (Apple M5 Max — 128GB Unified Memory)

### 1. Specialist LoRA Training (`Gemma 4 12B` — 48 LoRA Layers)

All benchmarks recorded on an **Apple M5 Max (128GB Unified RAM)** using `mlx_lm.tuner.trainer`:

| Sequence Length | Batch Size | Gradient Checkpointing | Token Throughput | Peak Memory | Swapping / Lag |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8,192 tokens** | `2` | **Enabled (`true`)** | **339.25 tokens/sec** | **99.58 GB** | **None (0% Swap)** |
| **4,096 tokens** | `2` | Enabled (`true`) | **410.12 tokens/sec** | **54.89 GB** | None (0% Swap) |

> **Critical Performance Discovery:**  
> When training multi-turn `agentic` data at `--max-seq-length 8192`, **gradient checkpointing (`grad_checkpoint: true`)** is mandatory. Without checkpointing, activation graphs swell to ~643 GB, triggering macOS virtual memory swapping to NVMe SSD and dropping throughput to `<10 tokens/sec`. With checkpointing enabled, peak RAM remains locked at **99.5 GB** running at **339.25 tokens/sec**.

### 2. MultiLoRA-MoE Inference & Dynamic Serving

| Metric | Measured Performance (M5 Max 128GB) |
| :--- | :--- |
| **Generation Throughput (BF16 Base + Top-2 Experts)** | **42 – 58 tokens/sec** |
| **Dynamic Gating Latency Overhead** | **< 1.2 ms per token** |
| **Active Memory Footprint (Serving)** | **~26.4 GB total** |

---

## 🚀 Quickstart & Installation

```bash
# 1. Clone the repository
git clone https://github.com/True2456/MLX_Train.git
cd MLX_Train

# 2. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install mlx mlx-lm pytest
```

---

## 🛠️ Training Specialist Adapters

Configuration files are provided in `config/`:
- `config/theory_lora.yaml` (`Expert 0`: Math & Security)
- `config/agentic_lora.yaml` (`Expert 1`: Native Tool Calling)
- `config/asm_systems_lora.yaml` (`Expert 4`: Low-Level Assembly & Decompilation)

### Recommended Production Training Command

```bash
python3 -m mlx_lm lora -c config/agentic_lora.yaml \
  --max-seq-length 8192 \
  --batch-size 2 \
  --val-batches 5 \
  --grad-checkpoint \
  --steps-per-report 1 \
  --iters 3000
```

### Critical MLX Apple Silicon Best Practices:
1. **Always enable `--grad-checkpoint`** when sequence lengths exceed 4,096 tokens.
2. **Never set `max_seq_length` smaller than prompt length**: If a prompt exceeds `max_seq_length`, `mlx_lm` truncates away the assistant completion tokens (`ntoks = 0`), producing a divide-by-zero error.

---

## 🧪 Testing & Production Verification

Run the comprehensive 18-test verification suite to validate router gating, residual boosting, and multimodal compatibility:

```bash
pytest tests/ -v
```

---

## 📚 Documentation & Technical Reports

- **Dual/Multi-LoRA MoE Technical Report:** [`docs/MATI_12B_DUAL_LORA_MOE_TECHNICAL_REPORT.md`](docs/MATI_12B_DUAL_LORA_MOE_TECHNICAL_REPORT.md)
- **Gemma 4 12B Training Plan:** [`docs/GEMMA12B_MOE_TRAINING_PLAN.md`](docs/GEMMA12B_MOE_TRAINING_PLAN.md)
