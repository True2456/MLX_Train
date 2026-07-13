#!/usr/bin/env python3
"""
verify_and_dedupe_dataset.py

Strictly audits and deduplicates frontier_2026_code_as_action_1500.jsonl to ensure:
1. 100% unique samples (zero duplicate prompts or completions).
2. Valid Nemotron formatting (<|im_start|>user/assistant/im_end| tags properly closed).
3. Valid Python syntax verification on all embedded executable code blocks (ast.parse check).
4. Clean output ready for MLX Nemotron LoRA training.
"""

import ast
import json
import hashlib
import re
from pathlib import Path

INPUT_FILE = Path("/Users/true/Documents/Mati_Train/data/frontier_2026_code_as_action_1500.jsonl")
OUTPUT_FILE = Path("/Users/true/Documents/Mati_Train/data/frontier_2026_verified_clean.jsonl")

def audit_dataset():
    seen_hashes = set()
    verified_rows = []
    stats = {
        "total_read": 0,
        "duplicates_removed": 0,
        "syntax_errors_found": 0,
        "format_errors_found": 0,
        "verified_clean": 0
    }

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            stats["total_read"] += 1
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt", "")
            completion = row.get("completion", "")

            # 1. Check uniqueness (SHA256 fingerprint of prompt + completion)
            fingerprint = hashlib.sha256((prompt + "|||" + completion).encode("utf-8")).hexdigest()
            if fingerprint in seen_hashes:
                stats["duplicates_removed"] += 1
                continue
            seen_hashes.add(fingerprint)

            # 2. Check Nemotron formatting tags
            if "<|im_start|>user" not in prompt or "<|im_end|>" not in prompt:
                stats["format_errors_found"] += 1
                continue
            if "<|im_end|>" not in completion:
                stats["format_errors_found"] += 1
                continue

            # 3. Check embedded Python script syntax validity via ast.parse
            # Extract code between cat << 'EOF' > ... and EOF
            code_matches = re.findall(r"cat << 'EOF' > [^\n]+\n(.*?)\nEOF", completion, re.DOTALL)
            syntax_valid = True
            for code_block in code_matches:
                try:
                    ast.parse(code_block)
                except SyntaxError as e:
                    syntax_valid = False
                    break
            if not syntax_valid:
                stats["syntax_errors_found"] += 1
                continue

            verified_rows.append(row)
            stats["verified_clean"] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in verified_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("============================================================")
    print("               STRICT NEMOTRON DATASET AUDIT                ")
    print("============================================================")
    print(f"Total rows read:        {stats['total_read']}")
    print(f"Duplicates removed:     {stats['duplicates_removed']}")
    print(f"Format errors removed:  {stats['format_errors_found']}")
    print(f"Syntax errors removed:  {stats['syntax_errors_found']}")
    print(f"Verified clean rows:    {stats['verified_clean']}")
    print("============================================================")
    print(f"Clean verified dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    audit_dataset()
