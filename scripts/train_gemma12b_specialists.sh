#!/usr/bin/env bash
# Train Gemma 4 12B theory LoRA, then agentic LoRA (one at a time).
# - Attn + MLP, all 48 layers, rank 16
# - max_seq_length 8192
# - Crash-safe: checkpoints every 50 iters; re-run this script to resume
# - Before each new forge segment, archives numbered ckpts → segN_* (no overwrite)
# - Metal ~499k resource crashes (~200–280 iters): use scripts/train_gemma12b_autorestart.sh
set -euo pipefail

FORGE="${HOME}/Documents/gemma-forge"
MATI="${HOME}/Documents/Mati_Train"
OUT="${MATI}/models/gemma12b"
LOG="${OUT}/train_specialists.log"
CFG="${MATI}/config/forge_gemma12b.json"
BASE="${OUT}/base_gemma4_shim"
REAL_BASE="${HOME}/.lmstudio/models/mlx-community/gemma-4-12B-it-bf16"
MAX_SEQ="${MAX_SEQ:-4096}"
THEORY_ITERS="${THEORY_ITERS:-2000}"
AGENTIC_ITERS="${AGENTIC_ITERS:-3000}"
FORCE="${FORCE:-0}"
# Set AGENTIC=1 to run agentic after theory. Default: theory only.
AGENTIC="${AGENTIC:-0}"

mkdir -p "${OUT}"

# Ensure gemma4 text shim exists (mlx_lm 0.31 lacks gemma4_unified)
if [[ ! -f "${BASE}/config.json" ]]; then
  echo "==> Building gemma4 load shim…"
  python3 - <<PY
import json, shutil
from pathlib import Path
src = Path("${REAL_BASE}")
out = Path("${BASE}")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
for p in src.iterdir():
    if p.name == "config.json":
        continue
    (out / p.name).symlink_to(p)
cfg = json.loads((src / "config.json").read_text())
cfg["model_type"] = "gemma4"
if "text_config" in cfg:
    cfg["text_config"] = dict(cfg["text_config"])
    cfg["text_config"]["model_type"] = "gemma4_text"
(out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print("shim ready", out)
PY
fi

exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "Gemma 12B specialist LoRA training"
echo "Started: $(date)"
echo "Base:    ${BASE}"
echo "Log:     ${LOG}"
echo "LoRA:    attn + mlp, 48 layers, rank 16"
echo "Context: ${MAX_SEQ}"
echo "Resume:  re-run this script after a crash (checkpoints every 50 iters)"
echo "Force:   FORCE=1 to ignore DONE markers"
echo "============================================================"

cd "${FORGE}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="${FORGE}/src${PYTHONPATH:+:$PYTHONPATH}"

OVERWRITE_FLAG=()
if [[ "${FORCE}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

# Before a new forge segment, rename prior numbered ckpts (0000050_…) → segN_…
# so resume cannot silently overwrite them. Leaves adapters.safetensors untouched.
archive_segment_ckpts() {
  local dir="$1"
  local script="${MATI}/scripts/archive_segment_checkpoints.py"
  if [[ ! -f "${script}" ]]; then
    echo "WARNING: missing ${script} — skipping checkpoint archive"
    return 0
  fi
  python3 "${script}" "${dir}"
}

# --- Theory data (instruction/output → prompt/completion) ---
THEORY_SRC="${MATI}/curated/specialists/gemma12b/theory/theory_gemma12b_train.jsonl"
THEORY_STEPS="${OUT}/theory_steps.jsonl"
if [[ ! -f "${THEORY_STEPS}" ]]; then
  echo ""
  echo "==> Preparing theory steps…"
  python3 - <<PY
import json
from pathlib import Path
src = Path("${THEORY_SRC}")
dst = Path("${THEORY_STEPS}")
n = 0
with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out = (r.get("output") or "").strip()
        if not out:
            continue
        fout.write(json.dumps({
            "prompt": f"<bos><|turn>user\n{r['instruction']}<turn|>\n<|turn>model\n",
            "completion": out,
        }, ensure_ascii=False) + "\n")
        n += 1
print(f"Wrote {n} theory steps → {dst}")
PY
fi

echo ""
echo "==> [1/2] THEORY LoRA (${THEORY_ITERS} iters, seq=${MAX_SEQ})…"
if [[ -f "${OUT}/theory_lora/DONE" && "${FORCE}" != "1" ]]; then
  echo "Theory already DONE — skipping."
else
  archive_segment_ckpts "${OUT}/theory_lora"
  forge train \
    -c "${CFG}" \
    --data "${THEORY_STEPS}" \
    --train-output "${OUT}/theory_lora" \
    --train-iters "${THEORY_ITERS}" \
    --max-seq-length "${MAX_SEQ}" \
    "${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}"
fi

echo ""
echo "==> Theory status:"
ls -lah "${OUT}/theory_lora" || true

echo ""
echo "==> [2/2] AGENTIC LoRA (${AGENTIC_ITERS} iters, seq=${MAX_SEQ})…"
if [[ "${AGENTIC}" != "1" ]]; then
  echo "Skipping agentic (set AGENTIC=1 to train). Theory-only mode."
elif [[ -f "${OUT}/agentic_lora/DONE" && "${FORCE}" != "1" ]]; then
  echo "Agentic already DONE — skipping."
else
  # Prefer length-packed steps (see scripts/pack_agentic_sequences.py)
  AGENTIC_PACKED="${MATI}/curated/specialists/gemma12b/agentic/train_steps_packed_${MAX_SEQ}.jsonl"
  AGENTIC_RAW="${MATI}/curated/specialists/gemma12b/agentic/train_steps.jsonl"
  if [[ -f "${AGENTIC_PACKED}" ]]; then
    AGENTIC_DATA="${AGENTIC_PACKED}"
    echo "Using packed agentic data: ${AGENTIC_DATA}"
  else
    AGENTIC_DATA="${AGENTIC_RAW}"
    echo "WARNING: packed file missing (${AGENTIC_PACKED}); using raw ${AGENTIC_DATA}"
    echo "  Rebuild with: python3 ${MATI}/scripts/pack_agentic_sequences.py --max-seq-length ${MAX_SEQ}"
  fi
  archive_segment_ckpts "${OUT}/agentic_lora"
  forge train \
    -c "${CFG}" \
    --data "${AGENTIC_DATA}" \
    --train-output "${OUT}/agentic_lora" \
    --train-iters "${AGENTIC_ITERS}" \
    --max-seq-length "${MAX_SEQ}" \
    "${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}"
fi

echo ""
echo "==> Agentic status:"
ls -lah "${OUT}/agentic_lora" || true

echo ""
echo "============================================================"
echo "ALL DONE: $(date)"
echo "  theory:  ${OUT}/theory_lora"
echo "  agentic: ${OUT}/agentic_lora"
echo "Re-run this script anytime to resume a crashed stage."
echo "============================================================"
