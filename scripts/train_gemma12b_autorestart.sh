#!/usr/bin/env bash
# Auto-restart Gemma 12B specialist training after Metal resource-limit crashes.
#
# Apple Metal caps ~499k resource objects per process. Long 8192 agentic runs
# hit that around ~200–280 continuous iters; quitting clears the table and
# forge resumes from adapters.safetensors. This wrapper loops until DONE.
#
# Example (agentic, same knobs as the inner script):
#   MAX_SEQ=8192 AGENTIC=1 AGENTIC_ITERS=3000 bash scripts/train_gemma12b_autorestart.sh
#
# Optional:
#   MAX_RESTARTS=200   # give up after N crashes (default 200)
#   SLEEP_SEC=20       # pause between restarts (default 20)
#   FORCE=1            # pass through to ignore DONE (dangerous)
set -uo pipefail

MATI="${HOME}/Documents/Mati_Train"
OUT="${MATI}/models/gemma12b"
INNER="${MATI}/scripts/train_gemma12b_specialists.sh"
MAX_RESTARTS="${MAX_RESTARTS:-200}"
SLEEP_SEC="${SLEEP_SEC:-20}"
# Default AGENTIC=1 here — this wrapper exists mainly for the agentic Metal loop.
export AGENTIC="${AGENTIC:-1}"
export MAX_SEQ="${MAX_SEQ:-8192}"
FORCE="${FORCE:-0}"

if [[ ! -x "${INNER}" && ! -f "${INNER}" ]]; then
  echo "ERROR: missing ${INNER}"
  exit 1
fi
chmod +x "${INNER}" 2>/dev/null || true

done_path() {
  if [[ "${AGENTIC}" == "1" ]]; then
    echo "${OUT}/agentic_lora/DONE"
  else
    echo "${OUT}/theory_lora/DONE"
  fi
}

is_retryable() {
  local code="$1"
  local log="${OUT}/train_specialists.log"
  # Ctrl-C / SIGTERM — stop looping
  if [[ "${code}" -eq 130 || "${code}" -eq 143 || "${code}" -eq 137 ]]; then
    return 1
  fi
  if [[ ! -f "${log}" ]]; then
    return 0
  fi
  # Prefer retrying known Metal / MLX transient failures
  if tail -n 80 "${log}" | grep -qE 'Resource limit \(499000\)|metal::malloc|METAL.*out of memory|Failed to allocate'; then
    return 0
  fi
  # Unknown nonzero exit: still retry (resume is safe); user can Ctrl-C
  return 0
}

attempt=0
echo "============================================================"
echo "Gemma 12B auto-restart trainer"
echo "Inner:   ${INNER}"
echo "Done when: $(done_path)"
echo "Max restarts: ${MAX_RESTARTS}  sleep: ${SLEEP_SEC}s"
echo "AGENTIC=${AGENTIC}  MAX_SEQ=${MAX_SEQ}  AGENTIC_ITERS=${AGENTIC_ITERS:-}  THEORY_ITERS=${THEORY_ITERS:-}"
echo "============================================================"

while true; do
  DONE="$(done_path)"
  if [[ -f "${DONE}" && "${FORCE}" != "1" ]]; then
    echo "Already complete: ${DONE}"
    exit 0
  fi

  attempt=$((attempt + 1))
  if [[ "${attempt}" -gt "${MAX_RESTARTS}" ]]; then
    echo "ERROR: hit MAX_RESTARTS=${MAX_RESTARTS} without DONE. Last adapters should still be resumable."
    exit 1
  fi

  echo ""
  echo "==> Attempt ${attempt}/${MAX_RESTARTS}  $(date)"
  set +e
  bash "${INNER}"
  code=$?
  set -e

  if [[ -f "$(done_path)" && "${FORCE}" != "1" ]]; then
    echo "DONE after attempt ${attempt}: $(done_path)"
    exit 0
  fi

  if [[ "${code}" -eq 0 ]]; then
    # Inner exited cleanly but DONE missing (e.g. AGENTIC=0 and theory already done)
    echo "Inner script exited 0 without $(done_path). Stopping."
    exit 0
  fi

  if ! is_retryable "${code}"; then
    echo "Non-retryable exit ${code} (interrupt?). Stopping."
    exit "${code}"
  fi

  echo "Crashed with exit ${code} (likely Metal resource limit). Resume in ${SLEEP_SEC}s…"
  sleep "${SLEEP_SEC}"
done
