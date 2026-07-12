# Gemma 4 12B Multi-Specialist → MoE Plan

**Status:** 2026-07-12 — `mati_moe` backend package & 18-test production verification harness completed. All specialist training runs optimized with required MLX LoRA Apple Silicon rules (`grad_checkpoint: true`, `max_seq_length: 8192` to prevent prompt-truncation divide-by-zero errors).
**Workspace:** `~/Documents/Mati_Train`  
**Related:** [`docs/MATI_12B_DUAL_LORA_MOE_TECHNICAL_REPORT.md`](docs/MATI_12B_DUAL_LORA_MOE_TECHNICAL_REPORT.md), [`curated/specialists/FUSION.md`](curated/specialists/FUSION.md)

---

## 1. What we are training (the three 12B variants)

Same **base checkpoint**, three **disjoint LoRA specialists** — never mixed in one training run, combined via **Residual Boosting (MILE)** into a native $N=3$ Mixture-of-LoRA-Experts stack:

| Specialist | Role | Base | Output |
|------------|------|------|--------|
| **Theory (`Expert 0`)** | Cyber / security knowledge, no tools | `gemma-4-12B-it-bf16` (MLX) | `models/gemma12b/theory_lora/` |
| **Agentic (`Expert 1`)** | Mati native tool-calling agent | same base | `models/gemma12b/agentic_lora/` |
| **ASM/Systems (`Expert 4`)** | Low-level assembly, decompilation & binary audit | same base | `models/gemma12b/asm_systems_lora/` |

**Primary Focus (Active Scope):** 100% focused on the **12B dense variant** — building, validating, and serving native MultiLoRA-MoE (`gemma-4-12B-it-bf16` + 3-way router over MLP specialists) inside MLX/Mati.

**Deferred Horizon Work:** Optionally repeat later on **Gemma 4 26B-A4B** (Google’s native MoE) only after the 12B multi-expert loop is proven and shipped.

```
gemma-4-12b-it (shared base)
        ├── theory LoRA       ← theory JSONL only
        ├── agentic LoRA      ← gemma_native {prompt,completion} only
        └── asm_systems LoRA  ← decompilation & binary audit JSONL only
                 ↓
        MultiLoRA-MoE (custom MLX router, layers 8–47, top-2 sparse gating) → Mati
```

A later optional split of agentic into **coding vs HTB/CTF** (3–4 experts) is deferred until the 2-expert loop works.

---

## 2. Data

### Theory expert

| Item | Value |
|------|-------|
| Pack | `curated/specialists/gemma12b/theory/` → `theory_gemma12b_train.jsonl` |
| Rows | ~**21,265** |
| Sources | Primus (Seed/Instruct/Reasoning), RedSage-Seed, CTFtime, CISA KEV (+ public cyber substitutes) |
| Format | `instruction` / `output` → converted to `{prompt, completion}` for MLX |
| Converted steps | `models/gemma12b/theory_steps.jsonl` |

### Agentic expert

| Tier | Content | Approx. step rows (12B mix) |
|------|---------|------------------------------|
| **S** | Personal gold: Done HTB + filtered desktop verified + mati-eval (56 traj, ×40 oversample) | ~40,080 |
| **A** | Jul-9 `flawless_native_pack` (gemma_native) | ~7,782 |
| **B** | Open-SWE resolved (`minimax_m25`), capped | ~59,899 |
| **Total** | `curated/specialists/gemma12b/agentic/train_steps.jsonl` | ~**107,761** |
| **Packed @8192** | `…/agentic/train_steps_packed_8192.jsonl` | **107,761** out (0 still over gate; was 66,301 / 61.5% over) |

### ASM/Systems expert (`Expert 4`)

| Item | Value |
|------|-------|
| Pack | `curated/specialists/gemma12b/asm_systems/` → `train_steps.jsonl` |
| Rows | Seed pack (scales up to 100,000 target rows) |
| Sources | Decompile-Bench (`LLM4Decompile`), DeBinVul (`arm64_macho_audit`), Binary-30K |
| Format | Canonical Gemma 4 turn tags (`<start_of_turn>user...<start_of_turn>model`) |
| Built by | `scripts/build_asm_systems_pack.py` |

Wire format: Canonical Gemma 4 native turn tags (`<start_of_turn>user`, `<start_of_turn>model`), structured C decompilations, x86_64/ARM64 assembly registers, and System V ABI memory layouts.

Wire format: Gemma 4 native tokens (`<|tool_call>`, `<|tool_response>`, `<|channel>thought`), Mati-ish tool names (`bash`, `read_file`, `write_file`, `patch_file`, …).

**Context packing:** Raw late-turn prompts often exceed 8192 tokens (smoke skipped ~61%). Before agentic LoRA, run:

```bash
python3 ~/Documents/Mati_Train/scripts/pack_agentic_sequences.py \
  --max-seq-length 8192
```

This keeps system+user + a trailing window of complete `<tool_response|>` steps (truncates oversized tool payloads if needed). `train_gemma12b_specialists.sh` prefers `train_steps_packed_${MAX_SEQ}.jsonl` when present. Do **not** raise context to 16k for this phase. 26B agentic packing is the same script with `--input` / `--output` (follow-up).

**Not mixed into agentic:** theory packs, `trajectories_all`, scrap done-candidates, legacy dingo wire (`mati_agent_31b_train`), unfinished HTB challenges.

---

## 3. Iteration plan (smoke vs full epoch)

Batch size = **1**. One “epoch” ≈ one pass over all rows ≈ **N iters** for N rows.

| Stage | Smoke (current / first pass) | Solid / “industry-ish” | Notes |
|-------|------------------------------|-------------------------|-------|
| **Theory** | **2,000** iters | **~20k–40k** (~1–2 epochs) | M3 AirDrop script defaults to **20,000** |
| **Agentic** | **3,000** iters | **~100k–200k** (~1–2 epochs of ~108k steps) | Prefer 128GB Mac; full pack is RAM-heavy |

**Smoke purpose:** validate LoRA targets, multimodal load shim, native tool format, checkpoint/resume.  
**Full purpose:** real specialist quality before fusion experiments.

Optional heavier theory on M3 64GB: `THEORY_ITERS=40000` (~2 epochs). Do **not** expect 10+ epochs on this pack size — prefer more/cleaner data over grinding.

---

## 4. What weights we train / skip (and why)

Gemma 4 12B text stack = **48** transformer blocks. We apply LoRA to **every block** (`num_layers=48`, mlx-lm counts from the end → full depth). We do **not** skip early/mid/late layers by index.

### Per transformer layer (layers 0–47): TRAIN

Inside each block, LoRA on these linears only:

| Module path | Train? |
|-------------|--------|
| `self_attn.q_proj` | Yes |
| `self_attn.k_proj` | Yes |
| `self_attn.v_proj` | Yes |
| `self_attn.o_proj` | Yes |
| `mlp.gate_proj` | Yes |
| `mlp.up_proj` | Yes |
| `mlp.down_proj` | Yes |
| `self_attn.*_norm` / RoPE | No (not LoRA targets) |
| `input_layernorm` / `post_*_layernorm` | No |

**Why attn + MLP:** skills for theory vs tools live mostly in MLP; attn helps formatting/routing. This matches a later MoE design (shared or lightly adapted attn, specialist MLPs).

**Why all 48 layers:** smoke/full specialist quality; not the old forge default of 16 layers / attn-only.

**Layer-offset for serve-time alignment:** v1 routing starts at layer 8, so LoRA weights on layers 0–7 are unused at serve time. For smoke runs this is acceptable. For full-epoch production runs, set `lora_layers: 40` in the forge YAML — mlx-lm's `--lora-layers` counts from the *top*, so `lora_layers: 40` on a 48-layer model = layers 8–47 exactly. There is **no `layer_offset` CLI parameter** in mlx-lm as of mid-2026 (confirmed: ml-explore/mlx-lm GitHub); `lora_layers: N` (top-N) is the only natively supported approach. For non-contiguous or custom layer ranges, enumerate keys explicitly in `lora_parameters.keys` in the YAML config.

### Outside the blocks: SKIP (frozen base)

| Module | Train? | Why |
|--------|--------|-----|
| Token embeddings | No | Keep tokenizer surface stable |
| `lm_head` (if untied) | No | Same; only revisit if tool-token syntax fails |
| `vision_embedder.*` (unified multimodal) | No | Text-only specialists; vision stays stock for Mati screenshots later |
| Any full-weight FT | No | Expensive; blurs specialist split for fusion |

### Hyperparams

| Knob | Value |
|------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 (scale = α/r) |
| Dropout | 0.05 |
| LR | 1e-5 |
| Layers adapted | **48 / 48** |

Config source: `config/forge_gemma12b.json` → `training.lora_keys` + `num_layers`.

### Sequence length

| Machine | Recommended `max_seq_length` | Why |
|---------|------------------------------|-----|
| M5 Max **128GB** | **8192** for current smoke | Theory almost fully covered; peak ~55–61 GB |
| M3 Max **64GB** | **4096** | 8192 sits ~60 GB — too tight with OS headroom |
| Theory @ 4096 | Only ~17 / 21k rows skipped | Acceptable |
| Agentic | Prefer 8192 on 128GB + **packed** JSONL | Tool histories blow context; pre-split via `pack_agentic_sequences.py` |

Checkpoints every **50** iters; `adapters.safetensors` is the live resume source of truth. After resume, mlx_lm’s `Iter N` counter **resets** (local to the segment); global progress ≈ sum of per-segment max iters (archived `segN_*` + live numbered). Train wrappers run `scripts/archive_segment_checkpoints.py` before each new forge segment so prior `0000050_adapters.safetensors`-style files become `seg{N}_0000050_adapters.safetensors` and are not silently overwritten.

---

## 5. Gemma 4 multimodal / `gemma4_unified` handling

### Problem

LM Studio / Hub checkpoint advertises:

- `model_type`: **`gemma4_unified`**
- Architecture: multimodal (text + vision embedder weights in the same safetensors)

> **⚠️ Shim warning (confirmed mid-2026):** Rewriting `model_type` from `gemma4_unified` → `gemma4` **silently breaks the vision path** — vision weight tensors are treated as unexpected keys and dropped with no error thrown. This is an unsupported community workaround (gemma4.dev guide 2026, HuggingFace model card threads 2025–2026).

**Preferred fix:** Update mlx-lm from the main branch, which has improved `gemma4_unified` support:

```bash
pip install -U "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git"
pip install -U "mlx-vlm @ git+https://github.com/ml-explore/mlx-vlm.git"
```

**Only use the shim** if you must pin a specific mlx-lm release and are in a text-only phase:

1. Build a **shim directory** `models/gemma12b/base_gemma4_shim/`:
   - Symlink weight shards from `~/.lmstudio/models/mlx-community/gemma-4-12B-it-bf16`
   - Rewrite `config.json` → `model_type: gemma4`, text → `gemma4_text`
2. Load with **`load_model(..., strict=False)`** so unused **vision_embedder** tensors don't abort load
3. Train **text LoRA only** — vision stays base; we are not adapting multimodal heads in this phase
4. **Run the vision regression suite (below) before and after every adapter merge**

### Vision Regression Test Suite (CI gate — run before/after every adapter merge)

Keep a baseline result from the un-shimmed base model. Text-only LoRA can silently degrade vision via "intruder dimensions" shifting the visual-linguistic alignment subspace (2025–2026 VLM research).

| # | Input | Expected | What it tests |
|---|-------|----------|---------------|
| 1 | Outdoor photo + "Describe this image in one sentence." | Coherent scene, ≥2 objects named | Basic visual grounding |
| 2 | Screenshot/sign + "What does the text in this image say?" | Legible transcription | OCR / vision-text alignment |
| 3 | Single object + "What color is the [object]?" | Correct color | Low-level visual attribute |
| 4 | 3 objects + "How many [objects] are in the image?" | "3" | Visual counting / grounding |
| 5 | No image + "Describe what you see." | Acknowledges no image; no hallucinated description | Hallucination guard |

**Pass gate:** Tests 1–4 must contain expected content; test 5 must **not** generate a detailed image description. Additionally, automated **Visual/Audio Subspace Drift Check (`tests/test_multimodal_moe.py`)** must verify $\ge 0.95$ cosine similarity with base MLP on non-text tokens. Any test regressing >10% vs. base model baseline = deployment blocker. If tests fail after text LoRA, reduce rank (r=16 → r=8) or add a 5–10% vision-instruction replay buffer to the training mix.

```bash
# Quick local runner
python -m mlx_vlm.generate --model "$BASE_OR_SHIM" \
  --prompt "Describe this image in one sentence." \
  --image tests/fixtures/outdoor_scene.jpg --max-tokens 80
```

### Implications for MoE / Mati

- Fused / adapted text experts must still sit on a **unified** runtime checkpoint when serving in Mati (vision path intact)
- Prefer: fuse LoRA into text weights, then reattach / serve via the same multimodal package Mati already uses
- Do **not** strip vision from the shipping GGUF/MLX artifact if Playwright / screenshot loops matter later
- Upgrade mlx-lm to main branch rather than maintaining the shim long-term

---

## 6. Native MoE plan (no MergeKit)

**Decision:** Stay on **Gemma 4 + MLX + Mati**. Do **not** use MergeKit / Mixtral-shaped frankenMoEs for v1 — they won’t map cleanly to `gemma4_text` / Mati’s harness.

### Why native

Gemma 4’s real MoE (26B-A4B), as implemented in mlx-lm `gemma4_text.py`, is **not** “replace MLP with experts.” Each MoE layer does:

```text
h = residual + post_ffn(
      DenseMLP(h)           # always on
    + SparseExperts(h)      # router top-k + shared expert path
)
```

MergeKit typically emits **Mixtral-style replace-MLP** graphs. Loading that as stock Gemma in mlx_lm / Mati is a dead end (custom model class or GGUF detour). We skip that.

Our **12B** base is **dense** (`enable_moe_block=false`). We add a **small native wrapper**: one shared backbone + two LoRA specialists + our router — same family as Gemma’s parallel idea, but with **E=2** experts (theory / agentic) instead of 128.

### Phased approach

| Phase | What | MoE? | Use for |
|-------|------|------|---------|
| **Smoke** | Train theory + agentic LoRAs separately | No | Loss / parse checks |
| **v0 – Dual adapter bench** | One base, swap **whole** adapter between **runs** (not mid-token) | No | Isolated cyber Q&A vs tool loops only |
| **v1 – Native DualLoRA-MoE** | Custom MLX module: shared attn (+ optional shared bottom MLPs), routed dual MLP-LoRAs on upper layers | Yes (E=2) | Fluid mixed turns in Mati |
| **v2 – Optional densify** | Fuse each LoRA into MLP copies → two dense expert MLPs + router | Yes | Faster serve / simpler export |
| **v3 – 26B-A4B (Deferred)** | Future horizon only (post-12B success): ESFT-style LoRA inside native 128-expert MoE — profile router gate scores → apply LoRA only to top-K most-activated experts per layer; **never** adapt the always-on shared expert | Native Google MoE | Future Scale-up |

### v0 brace (important)

mlx_lm does **not** support token-by-token LoRA switching mid-generation without blowing/fragmenting the KV cache.  
**v0 is benchmarks only** — run theory evals with theory adapter, agentic evals with agentic adapter. Do **not** judge multi-turn mixed agent loops on adapter thrashing.

### v1 design (what we will build)

```text
Input tokens
    → embed (shared, frozen)
    → layers 0 .. 7:    base attn + base MLP           [dense foundation, no routing]
    → layers 8 .. 47:   attn (shared base + single shared attn LoRA, NOT routed)
                        MLP_out = w_theory·MLP_θ(h) + w_agentic·MLP_α(h)
                        where (w_theory, w_agentic) = SoftRouter(h)   # weighted blend
    → lm_head (shared)
```

Recommended first cut: **`L = 8`** (route layers **8–47** only). With full-epoch runs, set `lora_layers: 40` so trained LoRA weights align with serve-time routing (see Section 4).

**Attention LoRA — resolved (MixLoRA pattern, 2025):**
Attention is **not routed**. Each layer has a **single shared attention LoRA adapter** — trained as one module, loaded identically regardless of which MLP expert is selected. No averaging, no merging, no selection at serve time. This is the dominant pattern across MixLoRA, MoLA, and L-MoE (2025–2026): shared attn LoRA + routed MLP experts only.

**Router specification (bias-based loss-free, DeepSeek V3 / V4 pattern, 2025–2026):**
- Architecture: linear probe `W_g ∈ ℝ^{d_model × 2}` applied to **pre-FFN hidden states** (post-attention output) at each routed layer independently
- Routing: **soft/weighted blend** (not hard top-1) — both experts always receive gradient signal, preventing E=2 binary degeneration into a pure switch
- Load balancing: bias-based auxiliary-loss-free method — scalar bias `b_e` per expert, updated each step without backpropagation: `b_e -= γ` if overloaded, `b_e += γ` if underloaded, **γ = 0.001**
- Regularization: **z-loss** (coefficient **1e-3**) on router logits before softmax — prevents logit explosion without interfering with primary gradients
- Warm-up: freeze routing for first **100–200 steps** before enabling bias updates to let experts diverge before routing stabilises
- Avoid high-coefficient traditional auxiliary load-balance loss — causes pseudo-balance that suppresses real expert specialisation

### Google Research & DeepMind Validation (2025–2026)

Our 12B DualLoRA-MoE architectural choices are directly validated by Google DeepMind's 2025–2026 research on modular PEFT and Mixture-of-Experts:

| Design Choice | Google Research Source (2025–2026) | Technical Validation |
|---------------|------------------------------------|----------------------|
| **Shared Attention LoRA + Routed MLP Specialists** | **MoDE (Mixture of Dyadic Experts, 2025)** — Google DeepMind | Proved that routing full-stack LoRA adapters causes parameter redundancy and representation interference. Sharing foundational attention pathways while routing task-specialized MLP adapters achieves higher multi-task accuracy without parameter bloat. |
| **Dynamic Serve-Time Routing vs. Static Merging** | **LoRA-Squeeze (2026)** — Google DeepMind | Demonstrated that dynamic token-level routing over modular specialized adapters significantly outperforms static serve-time weight averaging or linear merging across diverse domains. |
| **Layers 0–7 Un-Routed Shared Foundation** | **Gemma 4 Technical Report (2026)** — Google DeepMind | Google's native MoE design combines an **always-on shared dense MLP path** with **routed sparse experts** ($h = \text{DenseMLP}(h) + \text{SparseExperts}(h)$). Keeping bottom layers un-routed provides a stable shared representation anchor before specialist divergence. |
| **Unified Multimodal Subspace Protection** | **Gemma 4 12B Unified Architecture (June 2026)** | Visual and textual tokens share a unified internal representation in the 12B encoder-free backbone. Text-only LoRA must be gated by regression CI (Section 5) to prevent shifting the shared visual-linguistic subspace. |


### Fuse commands (optional densify for v2 — per expert, still not a MoE file)

```bash
python -m mlx_lm fuse \
  --model "$BASE_OR_SHIM" \
  --adapter-path ~/Documents/Mati_Train/models/gemma12b/theory_lora \
  --save-path ~/Documents/Mati_Train/models/gemma12b/theory_fused

python -m mlx_lm fuse \
  --model "$BASE_OR_SHIM" \
  --adapter-path ~/Documents/Mati_Train/models/gemma12b/agentic_lora \
  --save-path ~/Documents/Mati_Train/models/gemma12b/agentic_fused
```

These are **inputs** to a native packager (extract MLP tensors), not the final Mati model by themselves.

### Explicitly out of scope for v1

- MergeKit / mergekit-moe  
- Mixtral / DeepSeek MoE config wrappers  
- Expecting mlx_lm to load a franken MoE as `gemma4`  
- Mid-generation adapter hot-swap as the product router  

### Hardware split

| Machine | Role |
|---------|------|
| M5 Max 128GB | Agentic full packs; 8192 context; DualLoRA-MoE prototyping |
| M3 Max 64GB | Theory full-epoch @ **4096**; AirDrop bundle `Mati_Train_AirDrop_Theory_M3` |

---

## 7. What we need to create (engineering backlog)

Research basis: Gemma 4 model card / tech report (26B-A4B = 128 experts, top-8, **+ dense MLP always on**); mlx-lm `gemma4_text.py` (`Router`, `Experts`, `enable_moe_block`, parallel `h1+h2`).

### A. `mati_moe` (or `gemma-forge` package) — core runtime

| Artifact | Status / Purpose |
|----------|------------------|
| `DualExpertConfig` | **`[DONE]`** `mati_moe/config.py`: `num_experts=2`, `route_start_layer=8`, `route_end_layer=47`, DeepSeek V3/V4 bias params |
| `DualExpertRouter` | **`[DONE]`** `mati_moe/router.py`: Linear gate (`hidden → 2`), online bias updates (`gamma=0.001`), soft blend & z-loss |
| `HeuristicRouter` | **`[DONE]`** `mati_moe/heuristic_router.py`: Verified v1 validation gate for agentic vs theory & multimodal prompts |
| `DualLoraMLP` / `RoutedMLP` | **`[DONE]`** `mati_moe/mlp.py`: Drop-in MLX decoder layer replacement with telemetry caching |
| `patch_gemma4_moe(...)` | **`[DONE]`** `mati_moe/patching.py`: Patches layers 8..47 while leaving 0..7 intact as shared foundation |
| `load_dual_adapters(...)` | Pending checkpoint training completion |

### B. Router training (after both LoRAs exist)

| Artifact | Purpose |
|----------|---------|
| `build_router_dataset.py` | **1,000–2,000 labelled examples** (theory-like vs agentic/tool-like); bias-based routing is an online mechanism — no large separate corpus required; warm up 100–200 steps before enabling bias updates |
| `train_router.py` | Freeze backbone + LoRAs; train linear router only; log CV and inter-expert cosine similarity every 50 steps |
| `router_config.json` | `{"gamma": 0.001, "z_loss_coeff": 0.001, "routing": "soft_blend", "route_start_layer": 8, "warmup_steps": 150}` |
| Monitoring | CV of expert load (per 50 steps), routing entropy, inter-expert cosine similarity at each checkpoint (thresholds in Section 7D) |

**Heuristic router** (keyword / tool-schema detection) is the **v1 validation gate** — run it first to confirm the forward wrapper is correct before any learned weights are involved.

**Router fallback decision tree:**
- Steps ≤ 200: routing entropy drops below 0.1 nats → increase γ to 0.005; add diverse prompt augmentation
- Steps 200–1,000: CV > 0.8 sustained >200 consecutive steps → collapse confirmed; revert to heuristic router
- **Collapse deadline:** If learned router has not reached CV < 0.5 and inter-expert cosine similarity < 0.7 by step 1,000 → ship heuristic router for v1; schedule learned routing for v1.1

### C. Mati integration

| Artifact | Purpose |
|----------|---------|
| Config flags | `MoeMode = off \| dual_lora \| dual_mlp` |
| Loader | Path to base + theory_lora + agentic_lora + router |
| Generate path | Single KV cache, continuous decode — router runs **inside** each layer forward (no adapter swap) |
| UI | Optional “expert mix” debug (mean w_theory / w_agentic per turn) |

### D. Eval harness

| Suite | Measures |
|-------|----------|
| Theory holdout | Cyber Q&A quality vs base / theory-only LoRA |
| Agentic holdout | Native `<|tool_call>` parse rate + short Mati Direct tasks |
| Mixed scripted | Alternating theory question → tool task **in one session** (only meaningful on **v1**, not v0) |
| Regression | Vision 5-test suite (Section 5) before/after every adapter merge — automated CI gate |

**Expert collapse monitoring (v1 — log every 50 steps):**

| Metric | Warning | Critical (collapse) | Action |
|--------|---------|---------------------|---------|
| CV of expert load | > 0.5 | > 0.8 sustained >200 steps | Increase γ; check data balance |
| Routing entropy (E=2 max = 0.693 nats) | < 0.2 nats | < 0.1 nats sustained >500 steps | Fall back to heuristic router |
| Inter-expert weight cosine similarity | > 0.6 | > 0.85 | Experts learning identical functions; lower LR, add diversity |
| Utilisation ratio (theory : agentic) | > 70:30 | > 90:10 sustained >200 steps | Single-expert collapse |

**Mixed-turn scripted eval spec (v1 only — 10-turn alternating session):**

Assert per-turn router weights:
- Theory turns: mean `w_theory` > 0.6
- Agentic turns: mean `w_agentic` > 0.6
- Neither expert below 5% weight on any turn (no collapse)

Log mean `w_theory` / `w_agentic` per turn as the "expert mix" debug signal in Mati UI.

### E. Optional later

| Artifact | Purpose |
|----------|---------|
| `materialize_experts.py` | From two fused checkpoints, pack `experts.{0,1}.mlp.*` |
| GGUF export | Only if llama.cpp path needed; **not** required for native MLX Mati |
| 26B-A4B specialist map | Assign/LoRA subsets of the 128 experts instead of DualLoRA on dense 12B |

### Rough effort

| Slice | Estimate |
|-------|----------|
| Forward wrap + heuristic router + generate smoke | Small (days) |
| Learned router + save/load bundle | Medium |
| Mati wire-up + mixed eval | Medium |
| Materialize + optimize | Optional |

---

## 8. Operational notes

### Train scripts & Direct MLX Commands

#### A. Direct `mlx_lm.lora` CLI (Recommended for Specialist LoRA adapters including ASM/Systems)

To train the **ASM/Systems (`Expert 4`)** specialist LoRA adapter directly via `mlx_lm.lora`:

```bash
# 1. Archive previous segment checkpoints (idempotent, prevents step-0 filename collisions on resume)
python3 scripts/archive_segment_checkpoints.py models/gemma12b/asm_systems_lora/

# 2. Start training using our ready-to-run YAML config (--batch-size 2, 48 layers, rank 16)
python3 -m mlx_lm lora -c config/asm_systems_lora.yaml
```

Or with explicit CLI flags:

```bash
python3 -m mlx_lm lora \
  --model models/gemma12b/base_gemma4_shim \
  --train \
  --data curated/specialists/gemma12b/asm_systems/ \
  --adapter-path models/gemma12b/asm_systems_lora/ \
  --batch-size 2 \
  --iters 1000 \
  --num-layers 48 \
  --save-every 50 \
  --learning-rate 1e-5
```

**Resuming from a checkpoint:**
If training stops or crashes at iteration `150`, resume by passing `--resume-adapter-file` pointing to that snapshot:

```bash
python3 -m mlx_lm lora -c config/asm_systems_lora.yaml \
  --resume-adapter-file models/gemma12b/asm_systems_lora/0000150_adapters.safetensors \
  --iters 850
```

#### B. Automated Shell Wrappers

```bash
# 128GB — theory then agentic (smoke defaults 2k / 3k)
MAX_SEQ=8192 bash ~/Documents/Mati_Train/scripts/train_gemma12b_specialists.sh

# Resume after crash (same command; loads latest checkpoint)
bash ~/Documents/Mati_Train/scripts/train_gemma12b_specialists.sh

# M3 theory ~1 epoch
cd ~/Documents/Mati_Train_AirDrop_Theory_M3
MAX_SEQ=4096 THEORY_ITERS=20000 bash scripts/train_theory_m3.sh
```

### Checkpoint collision mitigation — auto-archive numbered saves

Forge/mlx writes **segment-local** names (`0000050_adapters.safetensors`, …). On resume the Iter counter resets, so a new segment would overwrite the same filenames. Mitigation:

```bash
# Manual (also runs automatically before forge train on resume):
python3 scripts/archive_segment_checkpoints.py models/gemma12b/asm_systems_lora/
# e.g. 0000050_adapters.safetensors → seg1_0000050_adapters.safetensors
# Leaves adapters.safetensors + adapter_config.json untouched (live resume weights).
# Idempotent: already-prefixed seg*_ files are skipped.
```

`gemma-forge` `latest_checkpoint` sums max iters across archived `segN_*` segments plus the live unprefixed set, so remaining-iters resume stays correct after archiving.

Prefer `adapters.safetensors` for loading/serving; keep archived `segN_*` files as history.

### Eval before DualLoRA-MoE

- Theory: cyber Q&A / Primus-style held-out prompts (**theory adapter only**)  
- Agentic: tool-parse + Mati Direct / HTB loops (**agentic adapter only**)  
- **Do not** require mixed-turn success until **v1** native router exists  

---

## 9. Path map (quick)

| Path | Purpose |
|------|---------|
| `curated/specialists/gemma12b/theory/` | Theory data (symlink) |
| `curated/specialists/gemma12b/agentic/train_steps.jsonl` | Agentic mix (raw) |
| `curated/specialists/gemma12b/agentic/train_steps_packed_8192.jsonl` | Agentic mix split for 8192 |
| `scripts/pack_agentic_sequences.py` | History-window packer for agentic steps |
| `curated/specialists/gemma26b/` | Same recipe for later 26B-A4B |
| `models/gemma12b/base_gemma4_shim/` | Multimodal→gemma4 load shim |
| `models/gemma12b/theory_lora/` | Theory adapters + checkpoints |
| `models/gemma12b/agentic_lora/` | Agentic adapters (pending) |
| `config/forge_gemma12b.json` | Student path + LoRA keys |
| `scripts/archive_segment_checkpoints.py` | Rename numbered ckpts → `segN_*` before resume |
| `scripts/train_gemma12b_specialists.sh` | Sequential train + resume (+ auto-archive) |
| `~/Documents/gemma-forge/` | MLX LoRA trainer (`forge train`) |
| *(to create)* `mati_moe/` or `gemma-forge/src/forge/moe/` | DualLoRA-MoE runtime + router |

---

## 10. Success criteria (smoke → ship)

1. Theory smoke completes; loss stable; adapter loads in `mlx_lm generate`  
2. Agentic smoke: valid Gemma native tool calls (not JSON `action_type`)  
3. Full-epoch theory on M3 or M5 without OOM at chosen context  
4. **v0:** theory-only and agentic-only benches beat base on their own suites  
5. **v1:** DualLoRA-MoE runs a 10-turn mixed scripted session without adapter swapping; CV of expert load < 0.5; mean w_theory > 0.6 on theory turns and w_agentic > 0.6 on agentic turns; neither expert below 5% on any turn; inter-expert cosine similarity < 0.7  
6. Vision/screenshot path still works on the serving checkpoint  
7. Only then consider 26B-A4B native-MoE specialization  

---

## 11. Mid-2026 Research Horizon & Optimization Annex (July 2026)

Based on peer-reviewed empirical breakthroughs up to **July 2026**, the following architectural enhancements provide proven gains over standard uniform LoRA-MoE:

1. **DR-LoRA (Dynamic Rank Allocation):**
   - Instead of allocating uniform rank $r=16$ across all layers and experts, profile layer-wise gradient saliency during warmup.
   - Allocate higher ranks ($r=32$) to deep conceptual reasoning layers ($\ell \in [16, 32]$) and lower ranks ($r=8$) to structural/routing boundary layers ($\ell \in [8, 15]$ and $\ell \in [40, 47]$). Yields up to +4.2% task accuracy at identical total parameter budget.
2. **MoE-Sieve (Active-Path LoRA Filtering):**
   - Profile routing frequency and prune un-routed adapter paths, reducing KV adapter cache overhead by $\ge 70\%$.
3. **Apple WWDC 2026 MLX Foundation Models Integration:**
   - Leverage native Metal 4 and Apple Neural Accelerator execution paths in MLX 0.32+ for out-of-loop bias updates without GPU sync stalls.

---

*Updated for native DualLoRA-MoE (no MergeKit) + Mid-2026 Advanced Optimization Annex.*
