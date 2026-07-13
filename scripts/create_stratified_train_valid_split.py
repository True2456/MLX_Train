#!/usr/bin/env python3
"""
create_stratified_train_valid_split.py

Performs a rigorous 90/10 stratified train/validation split across our 1,881
verified Code-as-Action trajectories.

Exports:
  - dataset_qwen3_train.jsonl (1,693 rows) & dataset_qwen3_valid.jsonl (188 rows)
  - dataset_nemotron_train.jsonl & dataset_nemotron_valid.jsonl
  - dataset_gemma4_train.jsonl & dataset_gemma4_valid.jsonl

And automatically sets up symlinks in data_nemotron_theory/ for Qwen 3.6 training.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path("/Users/true/Documents/Mati_Train/data/frontier_2026_verified_clean.jsonl")
OUTPUT_DIR = Path("/Users/true/Documents/Mati_Train/data")
THEORY_DIR = Path("/Users/true/Documents/Mati_Train/data_nemotron_theory")

def classify_row(row):
    text = (row["prompt"] + " " + row["completion"]).lower()
    if "bytecode" in text or "opcode" in text or "evm" in text or "load_fast" in text:
        return "bytecode"
    elif "push" in text or "ldrsh" in text or "rbp" in text or "decompile" in text:
        return "assembly"
    elif "ast" in text or "taint" in text or "visitor" in text:
        return "ast"
    elif "raft" in text or "consensus" in text or "split-brain" in text:
        return "distributed"
    elif "asyncio" in text or "deadlock" in text or "race condition" in text:
        return "concurrency"
    else:
        return "general"

def extract_parts(row):
    prompt = row["prompt"]
    completion = row["completion"]
    user_query = prompt.replace("<|im_start|>user\n", "").replace("<|im_end|>\n<|im_start|>assistant\n", "").strip()

    m_think = re.search(r"<think>\n(.*?)\n</think>", completion, re.DOTALL)
    think_text = m_think.group(1).strip() if m_think else ""

    m_cmd = re.search(r"Action Input:\s*(.*?)(?:\n<\|im_end\|>|$)", completion, re.DOTALL)
    cmd_text = m_cmd.group(1).strip() if m_cmd else ""

    return user_query, think_text, cmd_text

def format_nemotron(u, t, c):
    prompt = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"<think>\n{t}\n</think>\nLet's execute the diagnostic script.\n\nAction: run_command\nAction Input: {c}<|im_end|>"
    return {"prompt": prompt, "completion": completion}

def format_gemma4(u, t, c):
    prompt = f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n"
    completion = f'<think>\n{t}\n</think>\nLet\'s execute the diagnostic script.\n\n<|tool_call>call:bash{{cmd:<|"|>{c}<|"|>}}<tool_call|><end_of_turn>'
    return {"prompt": prompt, "completion": completion}

def format_qwen3(u, t, c):
    prompt = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    args_json = json.dumps({"cmd": c}, ensure_ascii=False)
    completion = f'<think>\n{t}\n</think>\nLet\'s execute the diagnostic script.\n\n<tool_call>\n{{"name": "bash", "arguments": {args_json}}}\n</tool_call><|im_end|>'
    return {"prompt": prompt, "completion": completion}

def main():
    random.seed(42)  # Deterministic stratified split
    buckets = defaultdict(list)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            buckets[classify_row(r)].append(r)

    train_rows = []
    valid_rows = []

    print("Performing 90/10 stratified split across topic buckets:")
    for cat, items in buckets.items():
        random.shuffle(items)
        split_idx = int(len(items) * 0.90)
        t_items = items[:split_idx]
        v_items = items[split_idx:]
        train_rows.extend(t_items)
        valid_rows.extend(v_items)
        print(f"  - {cat:15s}: {len(t_items):4d} train | {len(v_items):3d} valid")

    random.shuffle(train_rows)
    random.shuffle(valid_rows)

    print(f"\nTotal Split: {len(train_rows)} Train | {len(valid_rows)} Valid")

    def export_dataset(rows, name_prefix):
        n_file = OUTPUT_DIR / f"{name_prefix}_nemotron.jsonl"
        g_file = OUTPUT_DIR / f"{name_prefix}_gemma4.jsonl"
        q_file = OUTPUT_DIR / f"{name_prefix}_qwen3.jsonl"

        with open(n_file, "w", encoding="utf-8") as fn, \
             open(g_file, "w", encoding="utf-8") as fg, \
             open(q_file, "w", encoding="utf-8") as fq:
            for r in rows:
                u, t, c = extract_parts(r)
                fn.write(json.dumps(format_nemotron(u, t, c), ensure_ascii=False) + "\n")
                fg.write(json.dumps(format_gemma4(u, t, c), ensure_ascii=False) + "\n")
                fq.write(json.dumps(format_qwen3(u, t, c), ensure_ascii=False) + "\n")

        return q_file

    qwen_train_file = export_dataset(train_rows, "train")
    qwen_valid_file = export_dataset(valid_rows, "valid")

    # Setup symlinks in data_nemotron_theory/
    THEORY_DIR.mkdir(parents=True, exist_ok=True)
    sym_train = THEORY_DIR / "train.jsonl"
    sym_valid = THEORY_DIR / "valid.jsonl"

    sym_train.unlink(missing_ok=True)
    sym_valid.unlink(missing_ok=True)

    sym_train.symlink_to(qwen_train_file)
    sym_valid.symlink_to(qwen_valid_file)

    print(f"\nSUCCESS: Symlinked for Qwen 3.6 training:")
    print(f"  {sym_train} -> {qwen_train_file}")
    print(f"  {sym_valid} -> {qwen_valid_file}")

if __name__ == "__main__":
    main()
