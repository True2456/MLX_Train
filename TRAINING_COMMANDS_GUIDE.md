# MATI MULTILORA-MOE TRAINING & SERVING GUIDE (2026 EDITION)
**Hardware Target:** Apple Silicon M5 Max (128GB Unified Memory)  
**Backend:** Apple MLX (`mlx-lm`)

---

## Table of Contents
1. [Core MLX Training Best Practices for 128GB RAM](#1-core-mlx-training-best-practices-for-128gb-ram)
2. [Nemotron 120B Coordinator Training (Two-Stage Curriculum)](#2-nemotron-120b-coordinator-training-two-stage-curriculum)
3. [Qwen 3.6 / Qwen 2.5 Coder 35B Execution Worker Training](#3-qwen-36--qwen-25-coder-35b-execution-worker-training)
4. [Gemma 4 26B-A4B Hybrid Vision/Execution Worker Training](#4-gemma-4-26b-a4b-hybrid-visionexecution-worker-training)
5. [Launching the Mati MultiLoRA-MoE Server](#5-launching-the-mati-multilora-moe-server)

---

## 1. Core MLX Training Best Practices for 128GB RAM

Always include the following flags when training on Apple Silicon to prevent memory overflow, swap lag, and `NaN` loss errors:
* **`KMP_DUPLICATE_LIB_OK=TRUE`**: Prevents OpenMP library collision crashes on macOS.
* **`--batch-size 1`**: Keeps peak gradient memory inside physical RAM.
* **`--max-seq-length 4096`**: Prevents long prompts from truncating and erasing target completion tokens (`0/0 = NaN`).
* **`--grad-checkpoint`**: Trades minimal compute overhead to drop peak RAM footprint by ~25–35%.
* **`--num-layers 16`**: Targets the upper 16 transformer layers where domain reasoning and tool execution logic reside.

---

## 2. Nemotron 120B Coordinator Training (Two-Stage Curriculum)

Use **Nemotron-3-Super-120B-A12B** as your **Strategic Coordinator & Planner**. Train it in two stages: first on cybersecurity theory, then on 2026 Frontier Code-as-Action.

### Stage 1: Domain Knowledge Infusion (`theory_steps.jsonl`)
Teaches the model deep cybersecurity theory, Windows/AD internals, SAM hive analysis, and system architecture.

```bash
# 1. Link the Theory dataset as train.jsonl
ln -sf /Users/true/Documents/Mati_Train_AirDrop_Theory_M3/data/theory_steps.jsonl /Users/true/Documents/Mati_Train/data_nemotron_theory/train.jsonl

# 2. Run Stage 1 Theory LoRA Training
KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python -m mlx_lm lora \
  --model /Users/true/.lmstudio/models/mlx-community/Nemotron-3-Super-120B-A12B-MLX-6bit \
  --data /Users/true/Documents/Mati_Train/data_nemotron_theory \
  --train \
  --batch-size 1 \
  --num-layers 16 \
  --max-seq-length 4096 \
  --grad-checkpoint \
  --iters 1500 \
  --learning-rate 3e-6 \
  --adapter-path /Users/true/Documents/Mati_Train/models/nemotron120b_stage1_theory
```

### Stage 2: Frontier Code-as-Action Sharpening (`dataset_nemotron_1141.jsonl`)
Resumes from Stage 1 and sharpens the Coordinator to delegate tasks and emit structured diagnostic commands (`Action: run_command`).

```bash
# 1. Link the native Nemotron Code-as-Action dataset
ln -sf /Users/true/Documents/Mati_Train/data/dataset_nemotron_1141.jsonl /Users/true/Documents/Mati_Train/data_nemotron_theory/train.jsonl

# 2. Run Stage 2 Code-as-Action LoRA Training (Resuming from Stage 1)
KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python -m mlx_lm lora \
  --model /Users/true/.lmstudio/models/mlx-community/Nemotron-3-Super-120B-A12B-MLX-6bit \
  --data /Users/true/Documents/Mati_Train/data_nemotron_theory \
  --train \
  --resume-adapter-file /Users/true/Documents/Mati_Train/models/nemotron120b_stage1_theory/adapters.safetensors \
  --batch-size 1 \
  --num-layers 16 \
  --max-seq-length 4096 \
  --grad-checkpoint \
  --iters 1141 \
  --learning-rate 2e-6 \
  --adapter-path /Users/true/Documents/Mati_Train/models/nemotron120b_coordinator_final
```

---

## 3. Qwen 3.6 / Qwen 2.5 Coder 35B Execution Worker Training

Use **Qwen 35B-A3B MoE / Qwen2.5-Coder-32B** as your **High-Speed Execution Worker** (generating at 100+ tokens/sec inside your harness).

### Training Command (`data_nemotron_theory/train.jsonl` + `valid.jsonl`)
Teaches Qwen to inspect state, write executable Python/Bash scripts inside XML tool calls (`<tool_call>`), and verify solutions deterministically. Automatically evaluates against our **189 stratified validation sequences** every 100 iterations.

```bash
KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python -m mlx_lm lora \
  --model /Users/true/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit \
  --data /Users/true/Documents/Mati_Train/data_nemotron_theory \
  --train \
  --batch-size 4 \
  --num-layers 16 \
  --max-seq-length 4096 \
  --grad-checkpoint \
  --iters 423 \
  --steps-per-eval 100 \
  --learning-rate 3e-6 \
  --adapter-path /Users/true/Documents/Mati_Train/models/qwen3.6_36b_a3b_code_as_action_lora
```

---

## 4. Gemma 4 26B-A4B Hybrid Vision/Execution Worker Training

Use **Gemma 4 26B-A4B MoE** when your execution worker requires **multimodal vision** (screenshot analysis, UI layouts, diagrams) alongside bash script generation.

### Training Command (`dataset_gemma4_1141.jsonl`)
Teaches Gemma 4 to emit `<|tool_call>call:bash{cmd:...}<tool_call|>` commands formatted with native `<start_of_turn>` tokens.

```bash
# 1. Link the native Gemma 4 Code-as-Action dataset
ln -sf /Users/true/Documents/Mati_Train/data/dataset_gemma4_1141.jsonl /Users/true/Documents/Mati_Train/data_nemotron_theory/train.jsonl

# 2. Run Gemma 4 Execution Worker LoRA Training
KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/bin/python -m mlx_lm lora \
  --model /Users/true/.lmstudio/models/mlx-community/gemma-4-26b-a4b-moe-mlx-6bit \
  --data /Users/true/Documents/Mati_Train/data_nemotron_theory \
  --train \
  --batch-size 2 \
  --num-layers 16 \
  --max-seq-length 4096 \
  --grad-checkpoint \
  --iters 1141 \
  --learning-rate 3e-6 \
  --adapter-path /Users/true/Documents/Mati_Train/models/gemma26b_worker_lora
```

---

## 5. Launching the Mati MultiLoRA-MoE Server

Launch your OpenAI-compatible API harness (`http://127.0.0.1:8080/v1`) and control how tool calls are normalized.

### Launch Command
```bash
python3 /Users/true/Documents/Mati_Train/scripts/serve_mati_moe.py \
  --port 8080 \
  --adapters-root /Users/true/Documents/Mati_Train/models
```

### Profile Switching API (Request Parameter `tool_profile`)
When calling `/v1/chat/completions`, pass `"tool_profile"` in your JSON body to isolate parsing format:
* `"tool_profile": "auto"` (Default — recognizes all formats: ReAct, Gemma, Qwen XML, Qwen `✿FUNCTION✿`)
* `"tool_profile": "gemma"` (Strictly only normalizes Gemma `<|tool_call>` format)
* `"tool_profile": "qwen"` (Strictly normalizes Qwen XML and `✿FUNCTION✿` format)
* `"tool_profile": "openai"` (Strictly normalizes OpenAI/ReAct format)
