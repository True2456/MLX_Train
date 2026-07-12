#!/usr/bin/env python3
"""Build authentic multi-thousand ASM & Systems dataset from Hugging Face
for Expert 4 (asm_systems) fine-tuning.
Harvests C/C++, Assembly, pointer, struct, kernel, memory, and reverse engineering
trajectories from HuggingFace open code datasets.
"""

import json
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "curated" / "specialists" / "gemma12b" / "asm_systems"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYWORDS = [
    "c++", " c ", "assembly", "asm", "pointer", "memory", "struct",
    "kernel", "binary", "buffer", "x86", "arm", "decompile", "malloc",
    "free", "segfault", "heap", "stack", "register", "vtable", "bitwise"
]

def render_gemma4_native_prompt(instruction: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"{instruction.strip()}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

def build_dataset():
    rows = []
    seen_prompts = set()

    print("Loading low-level systems & assembly items from sahil2801/CodeAlpaca-20k...")
    ds_alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    for item in ds_alpaca:
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if not instruction or not output or len(output) < 30:
            continue
        inst_lower = f" {instruction.lower()} "
        if any(kw in inst_lower for kw in KEYWORDS):
            prompt = render_gemma4_native_prompt(instruction)
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                rows.append({
                    "prompt": prompt,
                    "completion": f"{output.strip()}\n",
                    "source": "hf:sahil2801/CodeAlpaca-20k",
                    "subset": "asm_systems_c_lowlevel"
                })

    print(f"Loaded {len(rows)} trajectories from CodeAlpaca. Now loading from Evol-Instruct-Code-80k-v1...")
    ds_evol = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    for item in ds_evol:
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if not instruction or not output or len(output) < 30:
            continue
        inst_lower = f" {instruction.lower()} "
        if any(kw in inst_lower for kw in KEYWORDS):
            prompt = render_gemma4_native_prompt(instruction)
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                rows.append({
                    "prompt": prompt,
                    "completion": f"{output.strip()}\n",
                    "source": "hf:nickrosh/Evol-Instruct-Code-80k-v1",
                    "subset": "asm_systems_evol_c"
                })
        if len(rows) >= 3700:
            break

    print(f"Total harvested ASM & Systems trajectories: {len(rows)}")

    train_rows = rows[:3500]
    valid_rows = rows[3500:3700] if len(rows) > 3500 else rows[:100]

    train_path = OUT_DIR / "train_steps.jsonl"
    valid_path = OUT_DIR / "valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for r in valid_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully wrote {len(train_rows)} training trajectories -> {train_path}")
    print(f"Successfully wrote {len(valid_rows)} validation trajectories -> {valid_path}")

if __name__ == "__main__":
    build_dataset()
