#!/usr/bin/env python3
"""
generate_frontier_2026_dataset.py

Generates 1,500+ 2026 Frontier-Level Code-as-Action (Program-Aided Agentic Execution)
trajectories designed to rival frontier reasoning and agentic models (Claude Opus,
DeepSeek V4-Pro, GLM-5.2, Qwen3-Coder).

Each sample features:
1. Multi-turn inspection -> diagnosis -> fix -> stress verification.
2. Advanced real-world domains:
   - Asyncio Event Loop Race Conditions & Deadlock Isolation
   - Cryptographic RSA Partial Key / Exponent Recovery
   - AST-Based Static Taint & Vulnerability Tracking
   - Distributed System State Transition / Consensus Invariant Debugging
   - High-Performance C-extension / Buffer Memory Integrity Checks
"""

import json
import random
from pathlib import Path

OUTPUT_DIR = Path("/Users/true/Documents/Mati_Train/data")
OUTPUT_FILE = OUTPUT_DIR / "frontier_2026_code_as_action_1500.jsonl"

def gen_asyncio_race(seed_id):
    worker_count = 8 + (seed_id % 8)
    iterations = 500 + (seed_id % 500)
    expected_sum = worker_count * iterations

    prompt = (
        f"<|im_start|>user\nOur distributed worker pool (`pool_{seed_id}.py`) suffers from a subtle race condition "
        f"under concurrency load. When running {worker_count} async workers incrementing a shared state counter "
        f"{iterations} times each, the final total frequently falls below the expected invariant ({expected_sum}). "
        "Write an executable diagnostic script to reproduce the concurrency leak, isolate the non-atomic state update, "
        "apply an `asyncio.Lock` or atomic pattern, and stress-test the fix.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    completion = (
        "<think>\n"
        "To debug an async race condition reliably, I must first construct a stress test that reproduces the lost updates "
        "when multiple coroutines interleave at yield points (`await asyncio.sleep(0)`). Once reproduced, I will introduce "
        "an explicit `asyncio.Lock` around the critical read-modify-write section and verify zero lost updates across 100 runs.\n"
        "</think>\n"
        "Let's write a diagnostic script `test_race.py` to reproduce the lost updates and verify our synchronized fix.\n\n"
        "Action: run_command\n"
        "Action Input: cat << 'EOF' > test_race.py\n"
        "import asyncio\n"
        "\n"
        "class UnsafeCounter:\n"
        "    def __init__(self):\n"
        "        self.val = 0\n"
        "    async def inc(self):\n"
        "        curr = self.val\n"
        "        await asyncio.sleep(0)  # Yield execution point causing interleaving\n"
        "        self.val = curr + 1\n"
        "\n"
        "class SafeCounter:\n"
        "    def __init__(self):\n"
        "        self.val = 0\n"
        "        self.lock = asyncio.Lock()\n"
        "    async def inc(self):\n"
        "        async with self.lock:\n"
        "            curr = self.val\n"
        "            await asyncio.sleep(0)\n"
        "            self.val = curr + 1\n"
        "\n"
        f"async def run_workers(counter, workers={worker_count}, iters={iterations}):\n"
        "    async def worker():\n"
        "        for _ in range(iters):\n"
        "            await counter.inc()\n"
        "    await asyncio.gather(*(worker() for _ in range(workers)))\n"
        "    return counter.val\n"
        "\n"
        "async def main():\n"
        "    unsafe_res = await run_workers(UnsafeCounter())\n"
        "    safe_res = await run_workers(SafeCounter())\n"
        f"    print(f'UNSAFE_RESULT: {{unsafe_res}} / {expected_sum}')\n"
        f"    print(f'SAFE_RESULT: {{safe_res}} / {expected_sum}')\n"
        f"    assert safe_res == {expected_sum}, 'Race condition still present in SafeCounter!'\n"
        "\n"
        "asyncio.run(main())\n"
        "EOF\n"
        "python3 test_race.py\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Observation:\nUNSAFE_RESULT: {int(expected_sum * 0.72)} / {expected_sum}\n"
        f"SAFE_RESULT: {expected_sum} / {expected_sum}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        f"The diagnostic script proved that uncoordinated interleaving dropped ~28% of updates, whereas wrapping the "
        f"read-modify-write transition in `asyncio.Lock` guaranteed 100% invariant consistency ({expected_sum}/{expected_sum}).\n"
        "</think>\n"
        f"By executing our concurrency stress test, we verified the root cause and proved the fix:\n"
        f"* **Unsafe Execution:** Suffered lost updates due to coroutine yielding between read and write.\n"
        f"* **Synchronized Fix (`asyncio.Lock`):** Exactly **{expected_sum} / {expected_sum}** updates preserved deterministically across all concurrent workers.<|im_end|>"
    )
    return {"prompt": prompt, "completion": completion}

def gen_ast_taint_analysis(seed_id):
    func_name = f"handler_{seed_id}"
    param_name = f"user_payload_{seed_id}"

    prompt = (
        f"<|im_start|>user\nWrite a static AST taint analysis script in Python to audit `app_{seed_id}.py` and detect "
        f"unvalidated data flow from request parameter `{param_name}` into SQL sink `cursor.execute()`. Execute the script "
        "to prove zero-day vulnerability detection.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    completion = (
        "<think>\n"
        "To perform rigorous static taint analysis without executing untrusted code, I will write an `ast.NodeVisitor` "
        "that tracks variable assignments originating from the tainted parameter source and flags any call to `cursor.execute` "
        "where an unescaped tainted variable or f-string interpolation is passed as the query argument.\n"
        "</think>\n"
        "Let's create an AST taint tracker and test it against a vulnerable SQL query pattern.\n\n"
        "Action: run_command\n"
        "Action Input: cat << 'EOF' > taint_scanner.py\n"
        "import ast\n"
        "\n"
        "class TaintVisitor(ast.NodeVisitor):\n"
        "    def __init__(self, tainted_source):\n"
        "        self.tainted_vars = {tainted_source}\n"
        "        self.vulnerabilities = []\n"
        "\n"
        "    def visit_Assign(self, node):\n"
        "        # Check if right-hand side uses tainted variables\n"
        "        for child in ast.walk(node.value):\n"
        "            if isinstance(child, ast.Name) and child.id in self.tainted_vars:\n"
        "                for target in node.targets:\n"
        "                    if isinstance(target, ast.Name):\n"
        "                        self.tainted_vars.add(target.id)\n"
        "        self.generic_visit(node)\n"
        "\n"
        "    def visit_Call(self, node):\n"
        "        if isinstance(node.func, ast.Attribute) and node.func.attr == 'execute':\n"
        "            if node.args:\n"
        "                for child in ast.walk(node.args[0]):\n"
        "                    if isinstance(child, ast.Name) and child.id in self.tainted_vars:\n"
        "                        self.vulnerabilities.append((node.lineno, child.id))\n"
        "        self.generic_visit(node)\n"
        "\n"
        "code = '''\n"
        f"def {func_name}({param_name}):\n"
        f"    raw_query = f\"SELECT * FROM users WHERE id = '{{{param_name}}}'\"\n"
        "    cursor.execute(raw_query)\n"
        "'''\n"
        "tree = ast.parse(code)\n"
        f"scanner = TaintVisitor('{param_name}')\n"
        "scanner.visit(tree)\n"
        "for line, var in scanner.vulnerabilities:\n"
        "    print(f'TAINT_VULNERABILITY: Line {line} sink executes tainted variable [{var}]')\n"
        "EOF\n"
        "python3 taint_scanner.py\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Observation:\nTAINT_VULNERABILITY: Line 4 sink executes tainted variable [raw_query]\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "The AST scanner tracked the propagation from parameter source to formatted string assignment (`raw_query`) and flagged "
        "the SQL sink at line 4.\n"
        "</think>\n"
        f"By executing static AST taint analysis, our scanner successfully tracked data flow from `{param_name}` -> `raw_query` "
        f"and flagged the SQL injection vulnerability at **Line 4**.<|im_end|>"
    )
    return {"prompt": prompt, "completion": completion}

def gen_rsa_wiener_attack(seed_id):
    p = 10007
    q = 10009
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 179424673 % phi
    if e % 2 == 0:
        e += 1

    prompt = (
        f"<|im_start|>user\nWe intercepted an RSA public key `(n={n}, e={e})` where the encryption exponent `e` is abnormally large "
        "relative to `n`. Execute a continued fractions / Wiener attack verification script in Python to recover the private exponent `d` "
        "and prove decryption.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    completion = (
        "<think>\n"
        "When `e` is very large, `d` is small (`d < 1/3 * n^(1/4)`), making the key susceptible to Wiener's attack via continued fraction "
        "convergents of `e/n`. I will write an executable Python script to compute the convergents and test `(e*d - 1) % phi == 0`.\n"
        "</think>\n"
        "Let's execute a Python script implementing continued fraction convergents to recover the RSA private exponent `d`.\n\n"
        "Action: run_command\n"
        "Action Input: cat << 'EOF' > solve_rsa.py\n"
        "from fractions import Fraction\n"
        "\n"
        f"n = {n}\n"
        f"e = {e}\n"
        "\n"
        "def continued_fractions(n, d):\n"
        "    while d:\n"
        "        q = n // d\n"
        "        yield q\n"
        "        n, d = d, n % d\n"
        "\n"
        "def convergents(cf):\n"
        "    p0, p1 = cf[0], cf[0]*cf[1] + 1\n"
        "    q0, q1 = 1, cf[1]\n"
        "    yield p0, q0\n"
        "    yield p1, q1\n"
        "    for q in cf[2:]:\n"
        "        p2 = q * p1 + p0\n"
        "        q2 = q * q1 + q0\n"
        "        yield p2, q2\n"
        "        p0, p1 = p1, p2\n"
        "        q0, q1 = q1, q2\n"
        "\n"
        "cf = list(continued_fractions(e, n))\n"
        "for k, d_cand in convergents(cf):\n"
        "    if k == 0 or d_cand % 2 == 0:\n"
        "        continue\n"
        "    if (e * d_cand - 1) % k == 0:\n"
        "        phi_cand = (e * d_cand - 1) // k\n"
        "        if n - phi_cand + 1 > 0:\n"
        "            print(f'RECOVERED_PRIVATE_EXPONENT_d: {d_cand}')\n"
        "            break\n"
        "EOF\n"
        "python3 solve_rsa.py\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Observation:\nRECOVERED_PRIVATE_EXPONENT_d: 5\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "Wiener's continued fraction expansion successfully identified the private exponent `d = 5`.\n"
        "</think>\n"
        f"By executing Wiener's continued fraction attack over `e/n`, we recovered the private exponent **`d = 5`**.<|im_end|>"
    )
    return {"prompt": prompt, "completion": completion}

def gen_consensus_log_invariant(seed_id):
    term = 10 + (seed_id % 20)
    commit_idx = 45 + (seed_id % 30)

    prompt = (
        f"<|im_start|>user\nIn our Raft consensus simulation (`node_{seed_id}.log`), a follower node reports term `{term}` "
        f"with commit index `{commit_idx}`. Write and execute a Python state-machine verifier to test whether applying out-of-order "
        "log entries violates the AppendEntries monotonicity invariant.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    completion = (
        "<think>\n"
        "Raft's Log Matching and Commit Monotonicity invariants dictate that `commitIndex` cannot decrease and a follower must reject "
        "entries with older terms. I will execute an automated state-machine verification script to test AppendEntries sequence logic.\n"
        "</think>\n"
        "Let's execute a Python verification script to audit state transition invariants.\n\n"
        "Action: run_command\n"
        "Action Input: cat << 'EOF' > verify_raft.py\n"
        "class RaftFollower:\n"
        f"    def __init__(self):\n"
        f"        self.current_term = {term}\n"
        f"        self.commit_index = {commit_idx}\n"
        "\n"
        "    def append_entries(self, leader_term, leader_commit):\n"
        "        if leader_term < self.current_term:\n"
        "            return False, 'REJECT_STALE_TERM'\n"
        "        if leader_commit < self.commit_index:\n"
        "            return False, 'VIOLATION_MONOTONIC_COMMIT'\n"
        "        self.commit_index = max(self.commit_index, leader_commit)\n"
        "        return True, 'SUCCESS'\n"
        "\n"
        "node = RaftFollower()\n"
        "ok1, msg1 = node.append_entries(leader_term=node.current_term, leader_commit=node.commit_index + 5)\n"
        "ok2, msg2 = node.append_entries(leader_term=node.current_term - 1, leader_commit=node.commit_index + 10)\n"
        "print(f'TEST_VALID_ADVANCE: {msg1}')\n"
        "print(f'TEST_STALE_TERM_REJECTION: {msg2}')\n"
        "EOF\n"
        "python3 verify_raft.py\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Observation:\nTEST_VALID_ADVANCE: SUCCESS\nTEST_STALE_TERM_REJECTION: REJECT_STALE_TERM\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "The state machine correctly advanced `commitIndex` on valid terms and rejected stale leader terms deterministically.\n"
        "</think>\n"
        "By executing the Raft state verifier, we proved that the follower node upholds the AppendEntries monotonicity "
        "and term rejection invariants.<|im_end|>"
    )
    return {"prompt": prompt, "completion": completion}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print("Generating 1,500+ 2026 Frontier-Level Code-as-Action trajectories...")
    for i in range(1, 381):
        rows.append(gen_asyncio_race(i))
        rows.append(gen_ast_taint_analysis(i))
        rows.append(gen_rsa_wiener_attack(i))
        rows.append(gen_consensus_log_invariant(i))

    random.shuffle(rows)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"SUCCESS: Generated {len(rows)} 2026 Frontier-Level Code-As-Action training rows at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
