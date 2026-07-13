#!/usr/bin/env python3
"""
generate_asm_code_as_action.py

Converts low-level Assembly & Binary Analysis problems (x86_64 and Apple Silicon ARM64)
into 400 executable 'Code-as-Action' trajectories where the model writes verification scripts
to trace registers, simulate stack frames, and prove binary vulnerabilities.
"""

import json
import random
from pathlib import Path

OUTPUT_DIR = Path("/Users/true/Documents/Mati_Train/data")
ASM_CAA_FILE = OUTPUT_DIR / "asm_code_as_action_400.jsonl"

def gen_arm64_sign_ext_overflow(seed_id):
    pkt_len = 0x8000 + (seed_id % 1000)
    # Sign extension of 16-bit negative integer in Python:
    sign_extended = (pkt_len - 0x10000) & 0xFFFFFFFF

    prompt = (
        f"<|im_start|>user\nWe are analyzing an ARM64 Apple Silicon assembly snippet:\n"
        f"    ldrsh   w9, [x8, #4]      // Load 16-bit length at offset 4\n"
        f"    str     w9, [sp, #20]\n"
        f"    ldr     w2, [sp, #20]     // Pass as size_t argument to _memcpy\n"
        f"If the packet length field at offset +4 holds the 16-bit hex value `0x{pkt_len:04x}`, "
        "write and execute a Python script to verify the exact 32-bit sign-extended length passed to `memcpy` and confirm whether overflow occurs.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    completion = (
        "<think>\n"
        "I will write an executable Python verification script to model `LDRSH` (Load Register Signed Half-word) on a 16-bit integer "
        f"`0x{pkt_len:04x}` and inspect the resulting unsigned 32-bit `size_t` value passed to `memcpy`.\n"
        "</think>\n"
        "Let's execute a Python script to verify ARM64 sign-extension behavior.\n\n"
        "Action: run_command\n"
        "Action Input: python3 -c '\n"
        f"val_16 = 0x{pkt_len:04x}\n"
        "# LDRSH sign-extends bit 15 across 32 bits\n"
        "signed_val = val_16 - 0x10000 if (val_16 & 0x8000) else val_16\n"
        "unsigned_size_t = signed_val & 0xFFFFFFFF\n"
        "print(\"SIGNED_VAL:\", signed_val)\n"
        "print(f\"MEMCPY_SIZE_T: 0x{unsigned_size_t:08x}\")\n"
        "assert unsigned_size_t > 0xFFFF\n"
        "print(\"OVERFLOW_CONFIRMED: TRUE\")\n"
        "'\n<|im_end|>\n"
        "<|im_start|>user\n"
        f"Observation:\nSIGNED_VAL: {pkt_len - 0x10000}\nMEMCPY_SIZE_T: 0x{sign_extended:08x}\nOVERFLOW_CONFIRMED: TRUE\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\nThe script confirmed that `LDRSH` sign-extends `0x{pkt_len:04x}` into `0x{sign_extended:08x}`, causing a catastrophic buffer overflow.\n</think>\n"
        f"By executing our ARM64 register simulation script, we verified that `LDRSH` sign-extends `0x{pkt_len:04x}` into the 32-bit unsigned `size_t` value **`0x{sign_extended:08x}` ({sign_extended} bytes)**, confirming **Stack Buffer Overflow**.<|im_end|>"
    )
    return {"prompt": prompt, "completion": completion}

def gen_x86_64_stack_frame_verify(seed_id):
    offset = 24 + (seed_id % 8) * 8
    val = 1337 + seed_id

    prompt = (
        f"<|im_start|>user\nWe have the following x86_64 stack frame setup:\n"
        f"    push    rbp\n"
        f"    mov     rbp, rsp\n"
        f"    sub     rsp, 64\n"
        f"    mov     QWORD PTR [rbp-{offset}], {val}\n"
        f"    mov     rax, QWORD PTR [rbp-{offset}]\n"
        "Write and execute a Python script to simulate this stack frame offset and verify the return value in `rax`.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    completion = (
        "<think>\n"
        "I will write an executable Python script to model the stack frame byte array (`rbp - 64`), write the 64-bit integer "
        f"`{val}` at offset `- {offset}`, and read it back into simulated `rax`.\n"
        "</think>\n"
        "Let's execute a stack frame verification script.\n\n"
        "Action: run_command\n"
        "Action Input: python3 -c '\n"
        "stack = bytearray(128)\n"
        "rbp = 128\n"
        f"target_idx = rbp - {offset}\n"
        f"val = {val}\n"
        "stack[target_idx:target_idx+8] = val.to_bytes(8, \"little\")\n"
        "rax = int.from_bytes(stack[target_idx:target_idx+8], \"little\")\n"
        "print(\"VERIFIED_RAX:\", rax)\n"
        "assert rax == val\n"
        "'\n<|im_end|>\n"
        "<|im_start|>user\n"
        f"Observation:\nVERIFIED_RAX: {val}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\nThe script simulated the x86_64 stack frame and verified `rax = {val}`.\n</think>\n"
        f"By executing our stack frame simulation script, we verified that dereferencing `[rbp-{offset}]` yields **`rax = {val}`**.<|im_end|>"
    )
    return {"prompt": prompt, "completion": completion}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print("Generating 400 verified ASM & Binary Reverse Engineering Code-as-Action rows...")
    for i in range(1, 201):
        rows.append(gen_arm64_sign_ext_overflow(i))
        rows.append(gen_x86_64_stack_frame_verify(i))

    random.shuffle(rows)

    with open(ASM_CAA_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"SUCCESS: Saved 400 ASM Code-as-Action trajectories to {ASM_CAA_FILE}")

if __name__ == "__main__":
    main()
