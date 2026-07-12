# MLX_Train: Specialist LoRA Adapter Training Toolkit for Apple Silicon

**MLX_Train** is a high-performance training toolkit and configuration suite built for fine-tuning **domain-specialist LoRA adapters** (`Theory`, `Agentic`, `ASM/Systems`) on Apple Silicon using **Apple MLX** (`mlx_lm`).

---

## 💻 Hardware Requirements (Apple Silicon M-Series)

Fine-tuning large language models (`Gemma 4 12B / 26B`) natively on macOS leverages Apple Silicon's unified memory architecture and Metal Execution Engine.

| Requirement | Minimum Specification | Recommended Production Setup |
| :--- | :--- | :--- |
| **System SoC** | Apple M1 / M2 / M3 / M4 / M5 Max or Ultra | **Apple M5 Max / M4 Max / M3 Max** |
| **Unified Memory (RAM)** | **64 GB** (up to `4096` sequence length) | **128 GB** (up to `8192` sequence length) |
| **Operating System** | macOS 14.0+ (Sonoma or Sequoia) | macOS 15.0+ |
| **MLX Version** | `mlx >= 0.22.0`, `mlx-lm >= 0.21.0` | Latest release |

---

## ⚡ Verified Training Benchmark Speeds (Apple M5 Max — 128GB Unified Memory)

Benchmarks recorded fine-tuning **Gemma 4 12B** across **48 LoRA target layers** (`rank: 16`, `alpha: 32`):

| Sequence Length | Batch Size | Gradient Checkpointing | Token Throughput | Iteration Time | Peak Memory | Swapping / Lag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8,192 tokens** | `2` | **Enabled (`true`)** | **339.25 tokens/sec** | **~47s / step** | **99.58 GB** | **None (0% Swap)** |
| **8,192 tokens (Expert 4 ASM)** | `4` | **Enabled (`true`)** | **317.51 tokens/sec** | **~15s / step** | **35.39 GB** | **None (Val Loss 0.043 @ 2500 steps)** |
| **4,096 tokens** | `2` | Enabled (`true`) | **410.12 tokens/sec** | ~20s / step | **54.89 GB** | None (0% Swap) |

---

## 🚨 Critical Apple Silicon Fine-Tuning Best Practices

When training specialist LoRA adapters using `mlx_lm lora`:

1. **Always Enable Gradient Checkpointing (`grad_checkpoint: true`):**
   * When training at `--max-seq-length 8192`, you **MUST** enable gradient checkpointing in your YAML config or CLI (`--grad-checkpoint`).
   * **Why:** Without checkpointing, storing 88-layer intermediate backpropagation activations requires **~643 GB virtual memory**, forcing macOS to page GPU buffers to NVMe SSD. That causes severe system lag and command buffer aborts. With checkpointing enabled, peak RAM stays locked at **~99.5 GB** with zero SSD swapping.

2. **Never Set `max-seq-length` Below Prompt Length (`8192` Required for Agentic):**
   * In multi-turn Agentic datasets where system prompts + tool schemas are ~5,000 tokens long, setting `max-seq-length: 4096` truncates away the assistant completion tokens (`ntoks = 0`).
   * **Why:** This triggers a divide-by-zero error in `mlx_lm.tuner.trainer`, producing corrupted loss/token counters and dropping real training density from **~12,410 tokens/step down to ~118 tokens/step**.

3. **Optimal Iteration & Coverage Targets by Specialist:**
   * **Theory (`Expert 0` | Math & Reasoning):** 2,000–3,000 steps (~25M tokens). Up to 10,000 steps (~1 full epoch on 21k samples).
   * **Agentic (`Expert 1` | Native Tool Calling):** 2,500–3,000 total steps (~25–30M tokens). **Do not train for 10,000 steps**—overfitting Rank-16 LoRA causes mode collapse.
   * **ASM & Systems (`Expert 4` | Low-Level Auditing):** 1,000 steps on curated seed sets.

---

## 🚀 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/True2456/MLX_Train.git
cd MLX_Train

# 2. Create virtual environment and install MLX
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm
```

---

## 📂 Configuration Files (`config/`)

Ready-to-use production fine-tuning configs are located in `config/`:
* [`config/agentic_lora.yaml`](config/agentic_lora.yaml) — Specialist for tool calling, structured output, and multi-step reasoning.
* [`config/theory_lora.yaml`](config/theory_lora.yaml) — Specialist for math, cybersecurity, and conceptual reasoning.
* [`config/asm_systems_lora.yaml`](config/asm_systems_lora.yaml) — Specialist for low-level assembly and binary analysis.

### Example Configuration (`config/agentic_lora.yaml`)
```yaml
model: "models/gemma12b/base_gemma4_shim"
train: true
data: "curated/specialists/gemma12b/agentic"
adapter_path: "models/gemma12b/agentic_lora"
num_layers: 48
batch_size: 2
iters: 3000
learning_rate: 1e-5
save_every: 100
grad_checkpoint: true
max_seq_length: 8192
lora_parameters:
  rank: 16
  alpha: 32
  dropout: 0.05
  scale: 10.0
```

---

## 🎯 Launching LoRA Training

Run training directly via `mlx_lm lora`:

```bash
python3 -m mlx_lm lora -c config/agentic_lora.yaml \
  --max-seq-length 8192 \
  --batch-size 2 \
  --val-batches 5 \
  --grad-checkpoint \
  --steps-per-report 1 \
  --iters 500
```

To resume training from a saved adapter checkpoint:
```bash
python3 -m mlx_lm lora -c config/agentic_lora.yaml \
  --resume-adapter-file models/gemma12b/agentic_lora/0000200_adapters.safetensors \
  --max-seq-length 8192 \
  --batch-size 2 \
  --val-batches 5 \
  --grad-checkpoint \
  --steps-per-report 1 \
  --iters 500
```
