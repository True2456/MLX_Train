#!/usr/bin/env python3
"""Interactive & Batch CLI Runner for Mati 12B MultiLoRA-MoE (N=3 Specialists).

Usage:
  python scripts/run_mati_moe.py --prompt "Explain CVE-2026-3312"
  python scripts/run_mati_moe.py --check-only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mati_moe import MatiMoEEngine


def main():
    parser = argparse.ArgumentParser(description="Mati 12B MultiLoRA-MoE CLI Runner")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Decompile assembly pushq %rbp; movq %rsi, %r14 into C code",
        help="Prompt to route through the N=3 MultiLoRA-MoE stack",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify on-disk availability of all 3 production specialist adapters",
    )
    args = parser.parse_args()

    engine = MatiMoEEngine()

    print("===================================================================")
    print("           MATI 12B MULTILORA-MOE SERVING ENGINE (N=3)             ")
    print("===================================================================\n")

    checks = engine.verify_checkpoints_ready("/Users/true/Documents/Mati_Train/models/gemma12b")
    print("1. ACTIVE SPECIALIST CHECKPOINT AUDIT:")
    all_ready = True
    for name, info in checks.items():
        status = "READY" if info["ready"] else "MISSING"
        print(f"   [{status}] {name.upper()}:")
        print(f"            Path: {info['path']}")
        print(f"            Size: {info['size_mb']} MB | Iteration: {info['selected_iter']} | Val Loss: {info['loss']}")
        if not info["ready"]:
            all_ready = False

    if args.check_only:
        return

    print("\n2. EXECUTING MULTILORA-MOE TURN:")
    result = engine.generate_turn(args.prompt)
    routing = result["routing"]
    w = routing["weights"]

    print(f"   Input Prompt: \"{args.prompt}\"")
    print(f"   -> Dominant Expert: {routing['dominant_expert'].upper()}")
    print(
        f"   -> Routing Weights: Theory={w['theory']*100:.1f}% | "
        f"Agentic={w['agentic']*100:.1f}% | ASM/Systems={w['asm_systems']*100:.1f}%"
    )
    print(f"   -> MoE-Sieve Fast-Path: {routing['sieved_expert'].upper()} (skip inactive experts)")
    print(f"   -> Turn Latency: {result['latency_ms']} ms | Status: {result['status']}")
    print("\n===================================================================")


if __name__ == "__main__":
    main()
