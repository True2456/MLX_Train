#!/usr/bin/env python3
"""Comprehensive 15-Question MultiLoRA-MoE Benchmark Evaluation Suite.

Evaluates specialized domain knowledge across:
1. Low-Level Systems & Assembly Engineering (ASM Systems Specialist)
2. Advanced Defensive Cyber Architecture & Cryptography (Theory Specialist)
3. Structured Tool Orchestration & JSON Schema Compliance (Agentic Specialist)

Verifies router classification accuracy across all 15 advanced domain problems
and optionally generates live comparative responses against base Gemma 4 12B.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mati_moe import MatiMoEEngine

COMPREHENSIVE_BENCHMARK = [
    # --- DOMAIN 1: LOW-LEVEL SYSTEMS & ASSEMBLY ENGINEERING (asm_systems) ---
    {
        "id": "ASM_ABI_01",
        "domain": "Systems & Assembly",
        "expected": "asm_systems",
        "prompt": "In the x86_64 System V AMD64 ABI, if a C struct contains an int32_t, a float, and a double (total 16 bytes), exactly which CPU registers are used to pass this struct by value?",
    },
    {
        "id": "ASM_ELF_02",
        "domain": "Systems & Assembly",
        "expected": "asm_systems",
        "prompt": "In an ELF64 binary header (Elf64_Ehdr), what exact byte offsets and field sizes correspond to e_entry, e_phoff, and e_shoff?",
    },
    {
        "id": "ASM_PAGE_03",
        "domain": "Systems & Assembly",
        "expected": "asm_systems",
        "prompt": "In an x86_64 4-level paging scheme (CR3 -> PML4 -> PDPT -> PD -> PT), which specific bits in a Page Table Entry (PTE) control the NX (No-Execute) bit and User/Supervisor privilege level?",
    },
    {
        "id": "ASM_STACK_04",
        "domain": "Systems & Assembly",
        "expected": "asm_systems",
        "prompt": "Write an x86_64 assembly sequence to align the stack pointer RSP to a 16-byte boundary before making a C function call when RSP is at an unknown offset.",
    },
    {
        "id": "ASM_SYSCALL_05",
        "domain": "Systems & Assembly",
        "expected": "asm_systems",
        "prompt": "What register holds the syscall number and which registers hold arguments 1 through 6 for a Linux x86_64 syscall instruction versus a Linux ARM64 svc #0 instruction?",
    },
    # --- DOMAIN 2: ADVANCED DEFENSIVE CYBER & CRYPTOGRAPHIC VERIFICATION (theory) ---
    {
        "id": "THEORY_TLS_01",
        "domain": "Defensive Cyber & Crypto",
        "expected": "theory",
        "prompt": "In the TLS 1.3 cryptographic key derivation schedule (RFC 8446), what is the exact HKDF sequence to derive the client_handshake_traffic_secret from the Early Secret and ECDHE Shared Secret?",
    },
    {
        "id": "THEORY_RACE_02",
        "domain": "Defensive Cyber & Crypto",
        "expected": "theory",
        "prompt": "Explain how a Time-of-Check to Time-of-Use (TOCTOU) race condition occurs in Unix file access verification (access followed by open) and show the secure POSIX alternative using openat with O_NOFOLLOW.",
    },
    {
        "id": "THEORY_SIDE_03",
        "domain": "Defensive Cyber & Crypto",
        "expected": "theory",
        "prompt": "How does Montgomery ladder scalar multiplication prevent timing side-channel attacks on elliptic curve scalar multiplication compared to a naive double-and-add algorithm?",
    },
    {
        "id": "THEORY_CSP_04",
        "domain": "Defensive Cyber & Crypto",
        "expected": "theory",
        "prompt": "Design a strict Content Security Policy (CSP) Level 3 HTTP header mitigation that uses nonce-based execution for inline scripts, restricts frame ancestors to self, and prevents object injection.",
    },
    {
        "id": "THEORY_OAUTH_05",
        "domain": "Defensive Cyber & Crypto",
        "expected": "theory",
        "prompt": "Explain why PKCE (Proof Key for Code Exchange) using S256 code challenge is mathematically required for public OAuth 2.1 clients even when using HTTPS.",
    },
    # --- DOMAIN 3: AGENTIC ORCHESTRATION & STRUCTURED TOOL CALLING (agentic) ---
    {
        "id": "AGENTIC_SCHEMA_01",
        "domain": "Agentic Tool Calling",
        "expected": "agentic",
        "prompt": 'Given two conflicting tool schemas query_sql_db(query: str) and search_logs(keyword: str, level: str), format a valid JSON tool call array to search for "error logs".',
    },
    {
        "id": "AGENTIC_NESTED_02",
        "domain": "Agentic Tool Calling",
        "expected": "agentic",
        "prompt": "Construct a strict JSON-RPC 2.0 request payload invoking cloud.deploy with nested configuration parameters for CPU limits and memory quotas.",
    },
    {
        "id": "AGENTIC_CHAIN_03",
        "domain": "Agentic Tool Calling",
        "expected": "agentic",
        "prompt": "Explain how an autonomous agent should handle a tool response containing a transient HTTP 429 rate limit error using exponential backoff jitter before retrying.",
    },
    {
        "id": "AGENTIC_MULTI_04",
        "domain": "Agentic Tool Calling",
        "expected": "agentic",
        "prompt": 'Format a parallel tool execution block calling fetch_metrics(hostname="node1") and fetch_metrics(hostname="node2") simultaneously.',
    },
    {
        "id": "AGENTIC_ERROR_05",
        "domain": "Agentic Tool Calling",
        "expected": "agentic",
        "prompt": 'Given a JSON schema requiring status: enum["PENDING", "ACTIVE", "CLOSED"], demonstrate how an agent validates and recovers from an unparsable model output.',
    },
]


def run_benchmark(verbose: bool = False):
    print("===================================================================")
    print("      MATI 12B MULTILORA-MOE COMPREHENSIVE 15-TASK BENCHMARK       ")
    print("===================================================================\n")

    engine = MatiMoEEngine()
    correct = 0
    total = len(COMPREHENSIVE_BENCHMARK)

    domain_stats = {
        "Systems & Assembly": {"correct": 0, "total": 0},
        "Defensive Cyber & Crypto": {"correct": 0, "total": 0},
        "Agentic Tool Calling": {"correct": 0, "total": 0},
    }

    for i, task in enumerate(COMPREHENSIVE_BENCHMARK, 1):
        turn = engine.generate_turn(task["prompt"])
        routing = turn["routing"]
        dominant = routing["dominant_expert"]
        weights = routing["weights"]

        passed = dominant == task["expected"]
        if passed:
            correct += 1
            domain_stats[task["domain"]]["correct"] += 1
        domain_stats[task["domain"]]["total"] += 1

        tag = "PASS" if passed else "FAIL"
        print(
            f"[{i:02d}/15] [{task['id']}] -> Routed: {dominant.upper():<11} "
            f"({weights[dominant]*100:5.1f}%) | Expected: {task['expected'].upper():<11} -> [{tag}]"
        )
        if verbose:
            print(f"        Prompt: {task['prompt'][:85]}...")
            print(
                f"        Telemetry: Theory={weights['theory']*100:.1f}% | "
                f"Agentic={weights['agentic']*100:.1f}% | ASM={weights['asm_systems']*100:.1f}%\n"
            )

    print("\n===================================================================")
    print("                    BENCHMARK RESULTS BY DOMAIN                    ")
    print("===================================================================")
    for dom, stats in domain_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f" * {dom:<28}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    print("-------------------------------------------------------------------")
    overall_acc = (correct / total) * 100
    print(f" OVERALL ROUTING ACCURACY     : {correct}/{total} ({overall_acc:.1f}%)")
    print("===================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Run 15-Question Comprehensive MoE Benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed telemetry per question")
    args = parser.parse_args()
    run_benchmark(verbose=args.verbose)


if __name__ == "__main__":
    main()
