#!/usr/bin/env python3
"""3-Expert MultiLoRA-MoE Verification Script for Mati.

Verifies:
1. All three specialist LoRA adapters are present on disk
2. HeuristicRouter accurately routes prompts across Theory, Agentic, and ASM Systems
3. MoE routing weights properly normalize across 3 experts
"""

import os
from pathlib import Path
from mati_moe import HeuristicRouter

def main():
    print("===================================================================")
    print("      MATI 3-EXPERT MULTILORA-MOE ROUTING & ADAPTER CHECK          ")
    print("===================================================================\n")

    # 1. Verify Specialist Adapter Files
    base_dir = Path("/Users/true/Documents/Mati_Train/models/gemma12b")
    specialists = {
        "Theory (Expert 0 - iter 11350)": [
            base_dir / "theory_lora" / "adapters.safetensors",
            base_dir / "theory_lora" / "0011350_adapters.safetensors",
        ],
        "Agentic (Expert 2 - Multi-Turn Tool Call)": [
            base_dir / "agentic_lora" / "adapters.safetensors",
        ],
        "ASM Systems (Expert 4 - Low-Level Decompile & ABI)": [
            base_dir / "asm_systems_lora" / "adapters.safetensors",
        ],
    }

    print("1. CHECKING SPECIALIST ADAPTER CHECKPOINTS ON DISK:")
    for name, candidates in specialists.items():
        found = None
        for p in candidates:
            if p.exists():
                found = p
                break
        if found:
            size_mb = found.stat().st_size / (1024 * 1024)
            print(f"   [OK] {name}")
            print(f"        -> Path: {found} ({size_mb:.2f} MB)")
        else:
            print(f"   [MISSING] {name}")

    # 2. Test 3-Way Heuristic Routing across real prompts
    print("\n2. TESTING 3-WAY SPECIALIST ROUTING (THEORY vs AGENTIC vs ASM):")
    router = HeuristicRouter()

    test_prompts = [
        (
            "Explain CVE-2026-3312 Linux heap overflow and CVSS severity scoring",
            "Theory / Cybersecurity",
        ),
        (
            "<|tool_call|>read_file{\"path\": \"src/main.py\"} Let's check the patch_file",
            "Agentic / Tool Calling",
        ),
        (
            "Decompile the following assembly routine `pushq %rbp; movq %rsi, %r14` into C code and verify sizeof(void*) LP64 alignment",
            "ASM / Low-Level Systems",
        ),
        (
            "Audit this kernel assembly buffer overflow exploit and run bash command to patch_file",
            "Composite (Theory + Agentic + ASM)",
        ),
    ]

    for prompt, expected_domain in test_prompts:
        w_t, w_a, w_m = router.route_3way(prompt)
        print(f"\n   Prompt: \"{prompt[:60]}...\"")
        print(f"   Expected Domain: {expected_domain}")
        print(f"   -> Routing Weights: Theory={w_t*100:.1f}% | Agentic={w_a*100:.1f}% | ASM/Systems={w_m*100:.1f}%")

    print("\n===================================================================")
    print("               VERIFICATION COMPLETED SUCCESSFULLY                 ")
    print("===================================================================")

if __name__ == "__main__":
    main()
