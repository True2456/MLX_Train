#!/usr/bin/env python3
"""
export_model_specific_datasets.py

Exports our 1,141 audited 2026 Frontier Code-as-Action trajectories into 3 dedicated,
100% native formatted datasets for:
1. Nemotron 120B (dataset_nemotron_1141.jsonl)
2. Gemma 4 26B (dataset_gemma4_1141.jsonl)
3. Qwen 3.6 / Qwen 2.5 Coder 35B (dataset_qwen3_1141.jsonl)
"""

import json
import re
from pathlib import Path

INPUT_FILE = Path("/Users/true/Documents/Mati_Train/data/frontier_2026_verified_clean.jsonl")
OUTPUT_DIR = Path("/Users/true/Documents/Mati_Train/data")

NEMOTRON_FILE = OUTPUT_DIR / "dataset_nemotron_1141.jsonl"
GEMMA4_FILE = OUTPUT_DIR / "dataset_gemma4_1141.jsonl"
QWEN3_FILE = OUTPUT_DIR / "dataset_qwen3_1141.jsonl"

def extract_core_parts(row):
    prompt = row["prompt"]
    completion = row["completion"]

    # Clean out formatting tags to get raw user query
    user_query = prompt.replace("<|im_start|>user\n", "").replace("<|im_end|>\n<|im_start|>assistant\n", "").strip()

    # Extract <think> block if present
    think_match = re.search(r"<think>\n(.*?)\n</think>", completion, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    # Extract command after Action Input:
    cmd_match = re.search(r"Action Input:\s*(.*?)(?:\n<\|im_end\|>|$)", completion, re.DOTALL)
    cmd_text = cmd_match.group(1).strip() if cmd_match else ""

    return user_query, think_text, cmd_text

def format_for_nemotron(user_query, think_text, cmd_text):
    prompt = f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    completion = (
        f"<think>\n{think_text}\n</think>\n"
        f"Let's execute the diagnostic script.\n\n"
        f"Action: run_command\n"
        f"Action Input: {cmd_text}<|im_end|>"
    )
    return {"prompt": prompt, "completion": completion}

def format_for_gemma4(user_query, think_text, cmd_text):
    prompt = f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
    completion = (
        f"<think>\n{think_text}\n</think>\n"
        f"Let's execute the diagnostic script.\n\n"
        f'<|tool_call>call:bash{{cmd:<|"|>{cmd_text}<|"|>}}<tool_call|><end_of_turn>'
    )
    return {"prompt": prompt, "completion": completion}

def format_for_qwen3(user_query, think_text, cmd_text):
    prompt = f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    args_json = json.dumps({"cmd": cmd_text}, ensure_ascii=False)
    completion = (
        f"<think>\n{think_text}\n</think>\n"
        f"Let's execute the diagnostic script.\n\n"
        f"<tool_call>\n{{\"name\": \"bash\", \"arguments\": {args_json}}}\n</tool_call><|im_end|>"
    )
    return {"prompt": prompt, "completion": completion}

def main():
    rows_read = 0
    nemotron_rows = []
    gemma4_rows = []
    qwen3_rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows_read += 1
            row = json.loads(line)
            user_query, think_text, cmd_text = extract_core_parts(row)

            nemotron_rows.append(format_for_nemotron(user_query, think_text, cmd_text))
            gemma4_rows.append(format_for_gemma4(user_query, think_text, cmd_text))
            qwen3_rows.append(format_for_qwen3(user_query, think_text, cmd_text))

    with open(NEMOTRON_FILE, "w", encoding="utf-8") as f:
        for r in nemotron_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(GEMMA4_FILE, "w", encoding="utf-8") as f:
        for r in gemma4_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(QWEN3_FILE, "w", encoding="utf-8") as f:
        for r in qwen3_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"SUCCESS: Exported {rows_read} rows into 3 native model formats:")
    print(f"  1. Nemotron 120B: {NEMOTRON_FILE}")
    print(f"  2. Gemma 4 26B:   {GEMMA4_FILE}")
    print(f"  3. Qwen 3.6 35B:  {QWEN3_FILE}")

if __name__ == "__main__":
    main()
