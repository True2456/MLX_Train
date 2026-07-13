#!/usr/bin/env python3
"""
build_code_as_action_dataset.py

Generates a multi-turn 'Code-As-Action' (Program-Aided Agentic Execution) dataset
for fine-tuning open-weight models (Nemotron, Qwen, Gemma) to solve problems by
writing and executing code in an interactive environment rather than guessing.
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("/Users/true/Documents/Mati_Train/data")
OUTPUT_FILE = OUTPUT_DIR / "code_as_action_steps.jsonl"

def build_trajectories():
    trajectories = []

    # Domain 1: SWE & Debugging - Binary Search Off-by-One Fix
    trajectories.append({
        "prompt": "<|im_start|>user\nWe have a buggy binary search implementation in search.py that enters an infinite loop when searching for targets not in the array. Write an executable Python script to test edge cases, reproduce the bug, and verify your fix.<|im_end|>\n<|im_start|>assistant\n",
        "completion": """<think>
To diagnose an infinite loop in binary search reliably, I should not just eyeball the code. I will write a test script that runs the search with a timeout on edge cases (target smaller than minimum, target larger than maximum, target missing inside range) and inspect the exact execution state.
</think>
Let's write a diagnostic script `test_search.py` that tests our binary search implementation against boundary edge cases.

Action: run_command
Action Input: cat << 'EOF' > test_search.py
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid  # BUG: Should be mid + 1
        else:
            right = mid - 1
    return -1

import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Infinite loop detected!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(1)

try:
    print("Testing missing element greater than all items...")
    res = binary_search([1, 3, 5, 7], 8)
    print("Result:", res)
except TimeoutError as e:
    print("CRASH:", e)
EOF
python3 test_search.py<|im_end|>
<|im_start|>user
Observation:
Testing missing element greater than all items...
CRASH: Infinite loop detected!<|im_end|>
<|im_start|>assistant
<think>
The execution confirmed the infinite loop when searching for `8` in `[1, 3, 5, 7]`.
Why did it loop? When `left = 2 (val=5)` and `right = 3 (val=7)`, `mid = (2+3)//2 = 2`.
Since `arr[2] < 8`, it sets `left = mid = 2`, repeating the loop forever.
Fix: update `left = mid + 1` when `arr[mid] < target`. Let's write the fixed version and run a full assertion test suite.
</think>
The bug is reproduced: setting `left = mid` causes an infinite loop when `left` and `right` are adjacent. Let's patch `search.py` and run our verification suite.

Action: run_command
Action Input: cat << 'EOF' > search.py
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9]
    assert binary_search(arr, 5) == 2
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 8) == -1
    assert binary_search(arr, 0) == -1
    print("ALL TESTS PASSED SUCCESSFULLY.")
EOF
python3 search.py<|im_end|>
<|im_start|>user
Observation:
ALL TESTS PASSED SUCCESSFULLY.<|im_end|>
<|im_start|>assistant
<think>
The verification script passed all boundary tests deterministically.
</think>
The off-by-one error has been fixed and verified via execution. `search.py` now correctly advances `left = mid + 1`, preventing infinite loops on missing elements.<|im_end|>"""
    })

    # Domain 2: Systems & Log Analysis - HTTP 500 Endpoint Aggregation
    trajectories.append({
        "prompt": "<|im_start|>user\nWe have an nginx access log at `/var/log/nginx/access.log`. Find the top 3 endpoint URIs producing HTTP 500 errors and their exact counts by executing a command or Python script.<|im_end|>\n<|im_start|>assistant\n",
        "completion": """<think>
Instead of guessing or trying to read the raw log into memory, I will execute a deterministic Python script to filter HTTP 500 status codes, extract the request path, aggregate frequencies, and print the top 3 sorted results.
</think>
Let's execute a Python script to parse `/var/log/nginx/access.log`, filter for HTTP 500 status responses, extract the requested endpoints, and compute the top 3 error generators.

Action: run_command
Action Input: python3 -c '
from collections import Counter
import re

counts = Counter()
log_pattern = re.compile(r"\"[A-Z]+\s+([^\s]+)\s+HTTP/.*\"\s+500\s+")

with open("/var/log/nginx/access.log", "r") as f:
    for line in f:
        m = log_pattern.search(line)
        if m:
            counts[m.group(1)] += 1

for uri, count in counts.most_common(3):
    print(f"{count}\t{uri}")
'<|im_end|>
<|im_start|>user
Observation:
142	/api/v2/checkout
89	/auth/callback
34	/dashboard/export<|im_end|>
<|im_start|>assistant
<think>
The executed code parsed the log deterministically and returned the exact top 3 endpoints failing with HTTP 500.
</think>
By executing log aggregation against the access log, we identified the top 3 endpoints generating HTTP 500 server errors:

1. `/api/v2/checkout` — 142 errors
2. `/auth/callback` — 89 errors
3. `/dashboard/export` — 34 errors<|im_end|>"""
    })

    # Domain 3: Security & CTF - Cryptographic LCG Seed Recovery
    trajectories.append({
        "prompt": "<|im_start|>user\nWe intercepted a ciphertext `4f6c6a7a4f61` encrypted with XOR against a linear congruential generator (LCG) where `seed = (seed * 1337 + 42) % 256`. Find the 1-byte initial seed and recover the plaintext flag by executing a solver script.<|im_end|>\n<|im_start|>assistant\n",
        "completion": """<think>
Since the modulus is 256, there are only 256 possible initial seeds (0 to 255). Rather than manually calculating modular arithmetic, I will write an executable Python script to brute-force all 256 seeds, decrypt the hex string, and filter for printable ASCII text starting with a recognizable prefix or readable string.
</think>
Let's write and execute a Python solver script to brute-force all 256 initial seed values of the LCG and inspect the decrypted plaintext candidates.

Action: run_command
Action Input: cat << 'EOF' > solve_lcg.py
import binascii

ct = binascii.unhexlify("4f6c6a7a4f61")

for seed in range(256):
    s = seed
    pt = []
    for byte in ct:
        s = (s * 1337 + 42) % 256
        pt.append(byte ^ s)
    try:
        decoded = bytes(pt).decode("ascii")
        if decoded.isalnum() or "flag" in decoded.lower():
            print(f"Seed {seed}: {decoded}")
    except UnicodeDecodeError:
        pass
EOF
python3 solve_lcg.py<|im_end|>
<|im_start|>user
Observation:
Seed 104: SECURE<|im_end|>
<|im_start|>assistant
<think>
The execution brute-forced all 256 states instantly and discovered that Seed `104` decrypts `4f6c6a7a4f61` to the plaintext `SECURE`.
</think>
By executing a deterministic brute-force solver over the 8-bit state space (`0-255`), we recovered the initial seed **104** and decrypted the ciphertext to the plaintext **`SECURE`**.<|im_end|>"""
    })

    # Domain 4: Algorithmic Math & Data Verification - SHA256 Collision Prefix
    trajectories.append({
        "prompt": "<|im_start|>user\nFind an ASCII string starting with `user_test_` whose SHA-256 hash begins with the hex characters `0000a`. Execute a script to find and prove the solution.<|im_end|>\n<|im_start|>assistant\n",
        "completion": """<think>
Finding a partial SHA-256 pre-image with 5 leading hex characters (`0000a`) requires testing roughly 16^5 / 2 ≈ 500,000 hashes. I will write an optimized Python script using `hashlib` to find the matching string deterministically.
</think>
Let's write and execute a Python script to search for an ASCII string prefix `user_test_` whose SHA-256 hash begins with `0000a`.

Action: run_command
Action Input: python3 -c '
import hashlib

prefix = "user_test_"
nonce = 0
while True:
    candidate = f"{prefix}{nonce}"
    h = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    if h.startswith("0000a"):
        print(f"FOUND: {candidate} -> {h}")
        break
    nonce += 1
'<|im_end|>
<|im_start|>user
Observation:
FOUND: user_test_314892 -> 0000a4d1f893427be5d87a21b3d9081e771a2e9c12b3c4f982143171887b33a0<|im_end|>
<|im_start|>assistant
<think>
The script found `user_test_314892` deterministically.
</think>
By executing our search script, we found the solution:
* **Input String:** `user_test_314892`
* **SHA-256 Hash:** `0000a4d1f893427be5d87a21b3d9081e771a2e9c12b3c4f982143171887b33a0`<|im_end|>"""
    })

    return trajectories

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectories = build_trajectories()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in trajectories:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"SUCCESS: Generated Code-as-Action dataset at {OUTPUT_FILE} ({len(trajectories)} multi-turn samples)")

if __name__ == "__main__":
    main()
