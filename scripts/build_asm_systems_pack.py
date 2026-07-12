#!/usr/bin/env python3
"""Build Low-Level Systems & Binary Reverse Engineering (Expert 4) dataset pack
for Gemma 4 12B DualLoRA / MultiLoRA MoE specialization.

Outputs (in curated/specialists/gemma12b/asm_systems/):
  - asm_systems_train.jsonl: Instruction/Alpaca format
  - asm_systems_messages.jsonl: Standard multi-turn messages format
  - train_steps.jsonl: Pre-rendered Gemma 4 native prompt/completion steps for mlx-lm
  - manifest.json: Metadata & recommended MLX LoRA hyperparameters
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "curated" / "specialists" / "gemma12b" / "asm_systems"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:20]


def render_gemma4_native_prompt(instruction: str) -> str:
    """Render instruction into canonical Gemma 4 native prompt string."""
    return (
        "<start_of_turn>user\n"
        f"{instruction.strip()}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


# Curated seed repository of low-level assembly, decompilation, binary analysis,
# and Apple Silicon ARM64 reverse engineering tasks.
CURATED_ASM_SYSTEMS_DATA = [
    {
        "instruction": (
            "Decompile the following x86_64 assembly routine into idiomatic C code and explain its memory behavior:\n\n"
            "check_auth:\n"
            "    push    rbp\n"
            "    mov     rbp, rsp\n"
            "    sub     rsp, 32\n"
            "    mov     QWORD PTR [rbp-24], rdi\n"
            "    mov     QWORD PTR [rbp-8], 0\n"
            "    mov     rax, QWORD PTR [rbp-24]\n"
            "    movzx   eax, BYTE PTR [rax]\n"
            "    test    al, al\n"
            "    je      .L2\n"
            "    mov     eax, 1\n"
            "    jmp     .L3\n"
            ".L2:\n"
            "    mov     eax, 0\n"
            ".L3:\n"
            "    leave\n"
            "    ret"
        ),
        "output": (
            "### Decompiled C Implementation\n\n"
            "```c\n"
            "#include <stdbool.h>\n"
            "#include <stdint.h>\n\n"
            "bool check_auth(const char *token) {\n"
            "    uint64_t status = 0; // [rbp-8]\n"
            "    if (token[0] != '\\0') {\n"
            "        return true;\n"
            "    }\n"
            "    return false;\n"
            "}\n"
            "```\n\n"
            "### Memory & Register Analysis\n"
            "1. **Stack Frame Setup:** Standard `push rbp; mov rbp, rsp; sub rsp, 32` allocates 32 bytes of local stack space.\n"
            "2. **Argument Passing:** The first pointer parameter (`rdi`) is spilled to `[rbp-24]`.\n"
            "3. **Dereference & Branch:** `movzx eax, BYTE PTR [rax]` loads the first byte (`token[0]`) with zero-extension. `test al, al` checks if the byte is null (`\\0`).\n"
            "4. **Return Value:** Sets `eax = 1` if non-null, or `eax = 0` if null before `leave; ret`."
        ),
        "source": "decompile_bench_curated",
        "subset": "x86_64_c_decomp",
    },
    {
        "instruction": (
            "Analyze the following ARM64 (Apple Silicon) assembly block from a Mach-O binary. Identify potential buffer overflow or pointer arithmetic vulnerabilities:\n\n"
            "_parse_packet:\n"
            "    sub     sp, sp, #64\n"
            "    stp     x29, x30, [sp, #48]\n"
            "    add     x29, sp, #48\n"
            "    str     x0, [sp, #24]\n"
            "    ldr     x8, [sp, #24]\n"
            "    ldrsh   w9, [x8, #4]\n"
            "    str     w9, [sp, #20]\n"
            "    ldr     w2, [sp, #20]\n"
            "    add     x0, sp, #4\n"
            "    ldr     x1, [sp, #24]\n"
            "    add     x1, x1, #8\n"
            "    bl      _memcpy\n"
            "    ldp     x29, x30, [sp, #48]\n"
            "    add     sp, sp, #64\n"
            "    ret"
        ),
        "output": (
            "### Vulnerability Diagnosis: Stack Buffer Overflow via Signed Integer Sign-Extension\n\n"
            "#### Root Cause Analysis\n"
            "1. **Signed Half-Word Load (`ldrsh`):** The instruction `ldrsh w9, [x8, #4]` loads a 16-bit signed integer from packet offset `+4` and sign-extends it into 32-bit register `w9`.\n"
            "2. **Implicit Cast to Size Parameter:** The value is stored to `[sp, #20]` and reloaded into `w2` (which serves as the `size_t n` argument for `_memcpy`).\n"
            "3. **Stack Buffer Overflow:** The destination buffer address passed in `x0` is `sp + #4` (a 16-byte stack array before offset `#20`). If the 16-bit length field at offset `+4` is negative (e.g., `0x8000`), sign-extension turns `w2` into a very large unsigned value when interpreted by `memcpy`, overflowing the 16-byte stack frame and clobbering the saved frame pointer (`x29`) and return address (`x30`) at `[sp, #48]`.\n\n"
            "#### Remediation\n"
            "Use unsigned 16-bit loading (`ldrh`) and enforce an explicit upper-bound check (`len <= 16`) prior to calling `memcpy`."
        ),
        "source": "debinvul_curated",
        "subset": "arm64_macho_audit",
    },
    {
        "instruction": (
            "Explain the C struct alignment and padding layout required to produce the following x86_64 field offsets seen in decompilation:\n"
            "- Offset 0x00: 1-byte flag\n"
            "- Offset 0x08: 8-byte heap pointer\n"
            "- Offset 0x10: 4-byte integer ID\n"
            "- Offset 0x14: 2-byte permissions bitmask\n"
            "- Total `sizeof(struct)` = 0x20 (32 bytes)"
        ),
        "output": (
            "### Struct Memory Layout & Padding Explanation\n\n"
            "To satisfy 64-bit ABI alignment rules where 8-byte pointers must reside at 8-byte boundaries (`alignof(void*) == 8`), the compiler introduces padding bytes:\n\n"
            "```c\n"
            "struct SecureNode {\n"
            "    uint8_t  flag;         // 0x00 (1 byte)\n"
            "    uint8_t  _pad0[7];     // 0x01..0x07 (7 bytes compiler padding)\n"
            "    void*    heap_ptr;     // 0x08 (8 bytes)\n"
            "    uint32_t node_id;      // 0x10 (4 bytes)\n"
            "    uint16_t perms;        // 0x14 (2 bytes)\n"
            "    uint8_t  _pad1[2];     // 0x16..0x17 (2 bytes trailing padding to reach 8-byte align)\n"
            "    // Total size: 24 bytes (0x18) or 32 bytes (0x20) if additional 8-byte tail pad exists\n"
            "};\n"
            "```\n\n"
            "#### Optimization\n"
            "Reordering fields by descending alignment (`heap_ptr`, `node_id`, `perms`, `flag`) reduces total struct size from 32 bytes to 16 bytes by eliminating internal padding gaps."
        ),
        "source": "systems_abi_curated",
        "subset": "struct_layout_alignment",
    },
    {
        "instruction": (
            "Reverse engineer the following disassembled function snippet and explain the bitwise hash algorithm implemented:\n\n"
            "hash_str:\n"
            "    mov     eax, 5381\n"
            ".L_loop:\n"
            "    movzx   edx, BYTE PTR [rdi]\n"
            "    test    dl, dl\n"
            "    je      .L_done\n"
            "    mov     ecx, eax\n"
            "    shl     eax, 5\n"
            "    add     eax, ecx\n"
            "    add     eax, edx\n"
            "    inc     rdi\n"
            "    jmp     .L_loop\n"
            ".L_done:\n"
            "    ret"
        ),
        "output": (
            "### Algorithm Identification: djb2 String Hash (Daniel J. Bernstein)\n\n"
            "#### Step-by-Step Decompilation\n"
            "1. **Seed Initialization:** `mov eax, 5381` initializes hash accumulator `hash = 5381`.\n"
            "2. **Byte Iteration:** `movzx edx, BYTE PTR [rdi]` reads character `c = *str`.\n"
            "3. **Hash Recurrence:**\n"
            "   - `mov ecx, eax; shl eax, 5; add eax, ecx` computes `(hash << 5) + hash` (equivalent to `hash * 33`).\n"
            "   - `add eax, edx` adds character byte `c`.\n"
            "4. **Pointer Increment:** `inc rdi` advances pointer to the next character until null byte (`test dl, dl; je .L_done`).\n\n"
            "#### Decompiled C Code\n"
            "```c\n"
            "unsigned long djb2_hash(const char *str) {\n"
            "    unsigned long hash = 5381;\n"
            "    int c;\n"
            "    while ((c = *str++)) {\n"
            "        hash = ((hash << 5) + hash) + c; // hash * 33 + c\n"
            "    }\n"
            "    return hash;\n"
            "}\n"
            "```"
        ),
        "source": "decompile_bench_curated",
        "subset": "crypto_hash_asm",
    },
    {
        "instruction": (
            "Analyze the following Apple Silicon ARM64 assembly snippet utilizing Pointer Authentication Codes (PAC). Explain how PAC protects the return address and identify if there is any strip/sign gadget misuse:\n\n"
            "_secure_dispatch:\n"
            "    paciasp\n"
            "    stp     x29, x30, [sp, #-32]!\n"
            "    mov     x29, sp\n"
            "    str     x0, [sp, #16]\n"
            "    ldr     x8, [sp, #16]\n"
            "    blr     x8\n"
            "    ldp     x29, x30, [sp], #32\n"
            "    autiasp\n"
            "    ret"
        ),
        "output": (
            "### Pointer Authentication Code (PAC) Inspection\n\n"
            "#### Mechanism Analysis\n"
            "1. **Prologue Protection (`paciasp`):** Computes a cryptographic signature over the return address (`x30`) combined with the stack pointer (`sp`) using the A-key, inserting the PAC tag into the unused upper bits of `x30` before pushing to stack.\n"
            "2. **Epilogue Verification (`autiasp`):** Authenticates the signed return address in `x30` against `sp`. If modified on stack, authentication invalidates the pointer bits, causing an immediate kernel segmentation fault on `ret`.\n\n"
            "#### Vulnerability Audit (Indirect Call Clobbering)\n"
            "While stack return flow is authenticated, the indirect function pointer call (`blr x8`) loaded from `[sp, #16]` lacks branch target authentication (`blraa` / `braa`). An attacker who overwrites local stack slot `[sp, #16]` can divert execution flow to any arbitrary instruction."
        ),
        "source": "debinvul_curated",
        "subset": "arm64_pac_audit",
    },
    {
        "instruction": (
            "Decompile the following x86_64 GCC `-O2` compiled switch-case jump table routine into C and explain the range bounds check:\n\n"
            "process_opcode:\n"
            "    sub     edi, 10\n"
            "    cmp     edi, 3\n"
            "    ja      .L_default\n"
            "    mov     rdi, QWORD PTR [rdi*8 + .L_table]\n"
            "    jmp     rdi\n"
            ".L_table:\n"
            "    .quad   .L_case10\n"
            "    .quad   .L_case11\n"
            "    .quad   .L_default\n"
            "    .quad   .L_case13"
        ),
        "output": (
            "### Decompiled Switch-Case Logic with Jump Table\n\n"
            "#### Decompiled C Implementation\n"
            "```c\n"
            "void process_opcode(unsigned int opcode) {\n"
            "    switch (opcode) {\n"
            "        case 10: handle_case10(); break;\n"
            "        case 11: handle_case11(); break;\n"
            "        case 13: handle_case13(); break;\n"
            "        default: handle_default(); break;\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "#### Assembly Optimization Analysis\n"
            "1. **Normalization (`sub edi, 10`):** Shifts the switch range from `[10..13]` down to `[0..3]`.\n"
            "2. **Unsigned Bounds Check (`cmp edi, 3; ja .L_default`):** Any original input `< 10` wraps around to a large unsigned integer (`> 3`) and correctly branches to `.L_default` via unsigned jump above (`ja`).\n"
            "3. **Jump Table Dereference:** Uses scaled indexed addressing `[rdi*8 + .L_table]` for $O(1)$ dispatch."
        ),
        "source": "decompile_bench_curated",
        "subset": "x86_64_jump_table",
    },
    {
        "instruction": (
            "Examine this ARM64 assembly loop utilizing NEON SIMD vector registers. Decompile the vectorized kernel into equivalent scalar C code:\n\n"
            "vector_add:\n"
            ".L_vector_loop:\n"
            "    subs    x2, x2, #4\n"
            "    ldr     q0, [x0], #16\n"
            "    ldr     q1, [x1], #16\n"
            "    fadd    v0.4s, v0.4s, v1.4s\n"
            "    str     q0, [x0, #-16]\n"
            "    b.gt    .L_vector_loop\n"
            "    ret"
        ),
        "output": (
            "### Decompiled SIMD Vector Kernel\n\n"
            "#### Decompiled C Code\n"
            "```c\n"
            "void vector_add(float *in_out, const float *addend, size_t count) {\n"
            "    // Processes 4 single-precision floats (128-bit NEON q-register) per iteration\n"
            "    for (size_t i = 0; i < count; i += 4) {\n"
            "        in_out[i + 0] += addend[i + 0];\n"
            "        in_out[i + 1] += addend[i + 1];\n"
            "        in_out[i + 2] += addend[i + 2];\n"
            "        in_out[i + 3] += addend[i + 3];\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "#### NEON Architectural Notes\n"
            "`ldr q0, [x0], #16` performs post-indexed 128-bit vector loading and auto-increments the pointer by 16 bytes. `fadd v0.4s` executes 4 simultaneous IEEE-754 single-precision float additions."
        ),
        "source": "decompile_bench_curated",
        "subset": "arm64_neon_simd",
    },
    {
        "instruction": (
            "Audit the following C++ virtual method call assembly pattern on x86_64 Linux. Explain how the vtable pointer (`_vptr`) is accessed and reconstruct the class interface:\n\n"
            "dispatch_event:\n"
            "    mov     rax, QWORD PTR [rdi]\n"
            "    mov     rax, QWORD PTR [rax + 16]\n"
            "    mov     esi, 1\n"
            "    jmp     rax"
        ),
        "output": (
            "### C++ Vtable Dispatch & Interface Reconstruction\n\n"
            "#### Assembly Mechanics\n"
            "1. **Object Instance (`rdi`):** In System V AMD64 C++ ABI, `rdi` points to the object instance `this`.\n"
            "2. **Vtable Dereference (`mov rax, [rdi]`):** The first 8 bytes of a polymorphic class instance hold `_vptr`, pointing to the virtual method table.\n"
            "3. **Slot Indexing (`mov rax, [rax + 16]`):** Loads the 3rd virtual function entry at byte offset `16` (`slot 2`).\n"
            "4. **Tail Call (`jmp rax`):** Passes `this` in `rdi` and second argument `1` in `esi` directly into the virtual method.\n\n"
            "#### Reconstructed C++ Interface\n"
            "```cpp\n"
            "class EventListener {\n"
            "public:\n"
            "    virtual ~EventListener() = default;           // Vtable slot 0 & 1 (deleting/complete dtor)\n"
            "    virtual void on_event(int event_type) = 0;    // Vtable slot 2 (+16 bytes)\n"
            "};\n\n"
            "void dispatch_event(EventListener* listener) {\n"
            "    listener->on_event(1);\n"
            "}\n"
            "```"
        ),
        "source": "systems_abi_curated",
        "subset": "cpp_vtable_layout",
    },
    {
        "instruction": (
            "Examine this stripped Linux kernel `ioctl` assembly handler. Identify the Time-of-Check to Time-of-Use (TOCTOU) race condition vulnerability:\n\n"
            "kernel_ioctl_handler:\n"
            "    mov     r12, rsi\n"
            "    mov     rdi, QWORD PTR [rsi]\n"
            "    cmp     rdi, 1024\n"
            "    ja      .L_err\n"
            "    # Simulate sleep or scheduling yield\n"
            "    call    schedule\n"
            "    mov     rax, QWORD PTR [r12]\n"
            "    mov     rdi, rax\n"
            "    call    kmalloc\n"
            "    ret"
        ),
        "output": (
            "### Vulnerability Audit: Kernel TOCTOU Double-Fetch\n\n"
            "#### Root Cause Diagnosis\n"
            "1. **First Fetch (Check):** `mov rdi, [rsi]` reads a user-space memory buffer length directly from user pointer `rsi` and verifies `len <= 1024`.\n"
            "2. **Context Switch Window:** During `call schedule` (or any preemption interval), another user thread can modify the memory value at `*rsi` to an arbitrarily large integer (`e.g., 0xFFFFFFFF`).\n"
            "3. **Second Fetch (Use):** `mov rax, [r12]` re-reads the length from the same user pointer *without re-validating the bounds check*, passing the mutated attacker-controlled length to `kmalloc`.\n\n"
            "#### Secure Remediation\n"
            "Copy the user structure once into a kernel-stack local variable using `copy_from_user` and perform all validation and allocation strictly on the immutable local kernel copy."
        ),
        "source": "debinvul_curated",
        "subset": "kernel_toctou_race",
    },
    {
        "instruction": (
            "Decompile the following RISC-V 64-bit (`RV64GC`) assembly routine into C code:\n\n"
            "str_length:\n"
            "    mv      a1, a0\n"
            ".L_loop:\n"
            "    lbu     a2, 0(a1)\n"
            "    addi    a1, a1, 1\n"
            "    bnez    a2, .L_loop\n"
            "    sub     a0, a1, a0\n"
            "    addi    a0, a0, -1\n"
            "    ret"
        ),
        "output": (
            "### Decompiled RISC-V 64 (`RV64GC`) Implementation\n\n"
            "#### Decompiled C Function\n"
            "```c\n"
            "size_t str_length(const char *str) {\n"
            "    const char *ptr = str;\n"
            "    while (*ptr != '\\0') {\n"
            "        ptr++;\n"
            "    }\n"
            "    return (size_t)(ptr - str);\n"
            "}\n"
            "```\n\n"
            "#### Register Architecture Walkthrough\n"
            "1. `mv a1, a0`: Copies base pointer `str` (`a0`) into scanning cursor `ptr` (`a1`).\n"
            "2. `lbu a2, 0(a1)`: Loads unsigned byte (`char`) from memory.\n"
            "3. `bnez a2, .L_loop`: Branches if character is non-zero.\n"
            "4. `sub a0, a1, a0; addi a0, a0, -1`: Computes difference minus 1 (accounting for post-increment after null byte hit) and returns length in `a0`."
        ),
        "source": "decompile_bench_curated",
        "subset": "riscv64_decomp",
    },
]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_pack() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_train: List[dict] = []
    rows_messages: List[dict] = []
    rows_steps: List[dict] = []

    for idx, item in enumerate(CURATED_ASM_SYSTEMS_DATA):
        instruction = item["instruction"].strip()
        output = item["output"].strip()
        fp = sha(f"{instruction}||{output}")

        # 1. Instruction / Alpaca format
        rows_train.append(
            {
                "instruction": instruction,
                "input": "",
                "output": output,
                "source": item["source"],
                "subset": item["subset"],
                "fingerprint": fp,
            }
        )

        # 2. Multi-turn conversational messages format
        rows_messages.append(
            {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "source": item["source"],
                "subset": item["subset"],
                "fingerprint": fp,
            }
        )

        # 3. Gemma 4 Native prompt/completion training steps format
        prompt_native = render_gemma4_native_prompt(instruction)
        rows_steps.append(
            {
                "prompt": prompt_native,
                "completion": output,
                "source": item["source"],
                "instance_id": f"asm_sys_{idx:04d}",
            }
        )

    n_train = write_jsonl(OUT_DIR / "asm_systems_train.jsonl", rows_train)
    n_msgs = write_jsonl(OUT_DIR / "asm_systems_messages.jsonl", rows_messages)
    n_steps = write_jsonl(OUT_DIR / "train_steps.jsonl", rows_steps)

    manifest = {
        "pack": "asm_systems_gemma12b",
        "expert_id": 4,
        "role": "Binary Reverse Engineering & Low-Level Systems Specialist",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows_train),
        "files": {
            "alpaca_format": str(OUT_DIR / "asm_systems_train.jsonl"),
            "messages_format": str(OUT_DIR / "asm_systems_messages.jsonl"),
            "gemma4_native_steps": str(OUT_DIR / "train_steps.jsonl"),
        },
        "recommended_train": {
            "base_model": "mlx-community/gemma-4-12b-it-bf16",
            "max_seq_length": 2048,
            "lora_ranks": {"core_layers": 32, "boundary_layers": 8},
            "batch_size": 4,
            "grad_accumulation_steps": 8,
            "iters": 3125,
            "notes": "Targeted Low-Level Assembly/Decompilation LoRA specialist for Mati MoE fusion.",
        },
    }

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Expert 4 ASM Systems dataset pack")
    parser.parse_args()
    manifest = build_pack()
    print(
        f"Built {manifest['pack']} ({manifest['row_count']} rows) -> {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
