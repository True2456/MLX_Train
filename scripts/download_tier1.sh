#!/usr/bin/env bash
# Download approved Tier 1 theory datasets (+ CTF substitutes for CyberSecurity-1M).
# Uses existing `hf auth login` / ~/.cache/huggingface/token.
set -euo pipefail

ROOT="${MATI_TRAIN_ROOT:-$HOME/Documents/Mati_Train}"

mkdir -p "$ROOT/theory/primus" "$ROOT/theory/redsage-seed" \
  "$ROOT/theory/substitutes-for-cybersecurity-1m/ctf"

download_one() {
  local id="$1" dest="$2"
  echo "==== $id ===="
  mkdir -p "$dest"
  hf download "$id" --repo-type dataset --local-dir "$dest"
  echo "OK  $id -> $dest"
}

download_one "RISys-Lab/RedSage-Seed" "$ROOT/theory/redsage-seed/RedSage-Seed"
download_one "trendmicro-ailab/Primus-Seed" "$ROOT/theory/primus/Primus-Seed"
download_one "trendmicro-ailab/Primus-Instruct" "$ROOT/theory/primus/Primus-Instruct"
download_one "trendmicro-ailab/Primus-Reasoning" "$ROOT/theory/primus/Primus-Reasoning"
download_one "justinwangx/CTFtime" "$ROOT/theory/substitutes-for-cybersecurity-1m/ctf/CTFtime"
download_one "justinwangx/CTFtime-unrolled" "$ROOT/theory/substitutes-for-cybersecurity-1m/ctf/CTFtime-unrolled"

echo "CyberSecurity-1M skipped — see theory/substitutes-for-cybersecurity-1m/README.md"
echo "Done."
