#!/usr/bin/env python3
"""Download authentic C/C++ Assembly & Decompilation datasets from Hugging Face
and compile thousands of real assembly/decompilation pairs for Expert 4 (ASM / Systems).
"""

import json
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "curated" / "specialists" / "gemma12b" / "asm_systems"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def render_gemma4_native_prompt(instruction: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"{instruction.strip()}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

def download_and_format():
    rows_steps = []
    print("Downloading authentic C/Assembly decompilation dataset from Hugging Face...")

    # Try downloading real C/Assembly decompilation datasets from HF
    datasets_to_try = [
        ("LLM4Binary/decompile-bench", "train", "asm", "c"),
        ("code_search_net", "c", "func_documentation_string", "func_code_string"),
    ]

    count = 0
    try:
        ds = load_dataset("LLM4Binary/decompile-bench", split="train")
        for idx, row in enumerate(ds):
            asm_code = row.get("input_asm") or row.get("asm") or row.get("instruction", "")
            c_code = row.get("output_c") or row.get("c") or row.get("output", "")
            if not asm_code or not c_code or len(asm_code) < 30 or len(c_code) < 20:
                continue
            instruction = f"Decompile the following assembly routine into idiomatic C code:\n\n```asm\n{asm_code.strip()}\n```"
            completion = f"### Decompiled C Implementation\n\n```c\n{c_code.strip()}\n```"
            prompt = render_gemma4_native_prompt(instruction)
            rows_steps.append({
                "prompt": prompt,
                "completion": completion,
                "source": "LLM4Binary/decompile-bench",
                "instance_id": f"hf_asm_{idx:05d}"
            })
            count += 1
            if count >= 3000:
                break
        print(f"Successfully harvested {count} authentic assembly decompilation pairs!")
    except Exception as e:
        print(f"Primary HF download note: {e}")

    out_path = OUT_DIR / "train_steps.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_steps:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows_steps)} real assembly/decompilation trajectories to {out_path}")

if __name__ == "__main__":
    download_and_format()
