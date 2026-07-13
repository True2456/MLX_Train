#!/usr/bin/env python3
"""
audit_final_dataset.py

Performs a strict deduplication audit (SHA-256 exact & normalized prompt check)
and reports topic distribution across the 1,521 frontier rows.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

FILE = Path("/Users/true/Documents/Mati_Train/data/frontier_2026_verified_clean.jsonl")

def get_topic(text):
    low = text.lower()
    if "bytecode" in low or "opcode" in low or "evm" in low or "load_fast" in low:
        return "Bytecode Disassembly & Reverse Engineering"
    elif "asyncio" in low or "deadlock" in low or "race condition" in low:
        return "Concurrency & Asyncio Distributed Debugging"
    elif "wiener" in low or "rsa" in low or "cryptanalysis" in low or "continued fraction" in low:
        return "Advanced Cryptanalysis & Mathematical Attack Verification"
    elif "ast" in low or "taint" in low or "static analysis" in low:
        return "AST Static Analysis & Vulnerability Detection"
    elif "raft" in low or "consensus" in low or "split-brain" in low:
        return "Distributed Systems Consensus & Protocol Verification"
    else:
        return "General Frontier Code-as-Action Problem"

def main():
    seen_hashes = set()
    unique_rows = []
    topic_counts = Counter()

    with open(FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # Hash normalized prompt + completion
            norm = (row["prompt"].strip() + "|||" + row["completion"].strip()).encode("utf-8")
            h = hashlib.sha256(norm).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_rows.append(row)
                topic_counts[get_topic(row["prompt"] + " " + row["completion"])] += 1

    # Overwrite clean file with strictly unique rows if any duplicates were pruned
    with open(FILE, "w", encoding="utf-8") as f:
        for r in unique_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"AUDIT COMPLETE:")
    print(f"  Total Unique Verified Rows: {len(unique_rows)}")
    print("\nTopic Breakdown across Frontier Categories:")
    for topic, count in topic_counts.most_common():
        pct = (count / len(unique_rows)) * 100
        print(f"  - {topic}: {count} rows ({pct:.1f}%)")

if __name__ == "__main__":
    main()
