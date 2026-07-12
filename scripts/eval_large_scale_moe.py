#!/usr/bin/env python3
"""Large-Scale 100-Prompt MultiLoRA-MoE Evaluation & Benchmark Suite.

Evaluates MoE router classification precision across 100 domain problems:
- 34 Systems & Assembly Engineering tasks (asm_systems)
- 33 Defensive Cybersecurity & Cryptography tasks (theory)
- 33 Agentic Tool Orchestration & Structured JSON Schema tasks (agentic)

Designed to verify end-to-end routing consistency for MatiApp client workloads.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mati_moe import MatiMoEEngine

# Generate 100 structured domain evaluation prompts
LARGE_SCALE_BENCHMARK = []

# --- 34 ASM & SYSTEMS ENGINEERING PROMPTS (asm_systems) ---
ASM_TEMPLATES = [
    "In the x86_64 System V AMD64 ABI, how are struct parameters passed when sizeof(struct) <= 16 bytes?",
    "Explain the exact layout of Elf64_Ehdr and what byte offset holds e_entry in an ELF64 binary header.",
    "Write an x86_64 assembly sequence to align the stack pointer RSP to a 16-byte boundary.",
    "Which bits in an x86_64 4-level Page Table Entry control the NX (No-Execute) flag and User/Supervisor access?",
    "Compare the register argument calling convention between Linux x86_64 syscall and Linux ARM64 svc #0.",
    "Demonstrate how to decompile a C function frame using rbp and rsp stack frame pointers.",
    "Explain struct padding and alignment rules in C for sizeof(void*) on 64-bit architectures.",
    "Write x86_64 assembly using pushq and popq to preserve callee-saved registers r12, r13, and r14.",
    "How does the CPU resolve virtual addresses via PML4, PDPT, PD, and PT tables?",
    "Explain the difference between leaq and movq instructions when computing memory addresses in x86_64.",
    "Describe the ELF section header table structure (Elf64_Shdr) and sh_offset resolving symbols.",
    "Demonstrate how stack unwinding works using DWARF Call Frame Information (CFI) instructions.",
    "Write an ARM64 assembly routine to allocate a 64-byte local stack buffer and store frame pointers.",
    "Explain how Model Specific Registers (MSRs) are read using rdmsr instruction on x86_64.",
    "Compare RISC-V register calling conventions (a0-a7) with System V AMD64 ABI (rdi, rsi, rdx).",
    "Explain the role of the RIP register when returning from a subq allocated stack frame.",
    "Write assembly code demonstrating how to pass floating point arguments in XMM0-XMM7 registers.",
    "Explain how the dynamic linker resolves symbols in the Global Offset Table (GOT) and PLT.",
    "Describe the binary header format of a Mach-O 64-bit executable load commands.",
    "Demonstrate how to inspect the stack frame of a segfault core dump using gdb assembly view.",
    "Write an x86_64 assembly loop that iterates over an array of 64-bit integers using rsi index.",
    "Explain the difference between callq near relative instruction versus indirect call register target.",
    "Describe how Thread Local Storage (TLS) segment register GS/FS base address is accessed in assembly.",
    "Write assembly code to perform atomic compare-and-swap using lock cmpxchg instruction.",
    "Explain how hardware breakpoints are configured using x86_64 Debug Registers DR0 through DR7.",
    "Describe how page faults (Interrupt 14) provide the faulting virtual address inside CR2 register.",
    "Demonstrate how to write inline assembly in C using GCC extended asm volatile syntax.",
    "Explain how to check CPUID flags for AVX2 and SSE4.2 vector extension support in assembly.",
    "Write an assembly routine to compute the strlen of a null-terminated string without C library calls.",
    "Explain how kernel stack switching works during a syscall transition via TSS Privilege Level 0 stack.",
    "Describe the purpose of the red zone 128-byte area below RSP in the x86_64 System V ABI.",
    "Write an x86_64 routine that allocates dynamic memory on the stack using subq rsp.",
    "Explain the differences between flat memory segmentation and protected mode descriptor tables.",
    "Demonstrate how to trace system calls emitted by a binary executable using strace and ptrace.",
]
for i, p in enumerate(ASM_TEMPLATES, 1):
    LARGE_SCALE_BENCHMARK.append({
        "id": f"ASM_{i:02d}",
        "domain": "Systems & Assembly",
        "expected": "asm_systems",
        "prompt": p,
    })

# --- 33 DEFENSIVE CYBERSECURITY & CRYPTOGRAPHY PROMPTS (theory) ---
THEORY_TEMPLATES = [
    "Explain how parameterized queries prevent SQL injection attacks in web applications.",
    "In the TLS 1.3 key derivation schedule RFC 8446, what is the sequence to derive client_handshake_traffic_secret?",
    "Explain how a TOCTOU race condition occurs in Unix file access and show the secure POSIX openat mitigation.",
    "How does Montgomery ladder scalar multiplication prevent timing side-channel attacks on elliptic curves?",
    "Design a strict Content Security Policy (CSP) Level 3 HTTP header using nonce-based script execution.",
    "Explain why PKCE using S256 code challenge is mathematically required for OAuth 2.1 public clients.",
    "Describe how Certificate Transparency log merkle trees prevent rogue CA certificates.",
    "Explain how HMAC-SHA256 authenticates message integrity against length extension attacks.",
    "Compare AES-GCM authenticated encryption Galois MAC against AES-CBC padding oracle vulnerabilities.",
    "Explain how SameSite=Strict cookies mitigate Cross-Site Request Forgery (CSRF) in browser sessions.",
    "Describe how Address Space Layout Randomization (ASLR) mitigates deterministic memory reuse attacks.",
    "Explain how kernel stack canaries detect buffer overflow corruption before function return.",
    "Explain how Subresource Integrity (SRI) hashes protect HTML pages from compromised CDN assets.",
    "Describe the cryptographic handshake of WireGuard Noise protocol framework IK pattern.",
    "Explain how perfect forward secrecy (PFS) ensures past sessions remain encrypted if private keys leak.",
    "Describe how WebAuthn FIDO2 public key challenge-response eliminates credential phishing attacks.",
    "Explain how Cross-Origin Resource Sharing (CORS) preflight OPTIONS requests restrict cross-origin APIs.",
    "Describe how secure boot measures kernel image signatures against UEFI db/dbx hardware roots of trust.",
    "Explain how password hashing with Argon2id resists GPU/ASIC brute-force dictionary attacks.",
    "Describe how TLS session resumption works using cryptographic session tickets and PSK.",
    "Explain how DNS over HTTPS (DoH) protects domain lookup privacy against path interception.",
    "Describe the defensive architecture of role-based access control (RBAC) vs attribute-based (ABAC).",
    "Explain how zero-knowledge proofs (ZKP) verify statements without revealing underlying secrets.",
    "Describe how input sanitization and context-aware output encoding prevent Cross-Site Scripting (XSS).",
    "Explain how kernel memory protections like SMEP and SMAP prevent executing user-space code.",
    "Describe how mutual TLS (mTLS) enforces cryptographically verifiable client identity in microservices.",
    "Explain how OAuth 2.0 Token Exchange RFC 8693 enables secure downstream service impersonation.",
    "Describe how security headers like HSTS enforce mandatory HTTPS transport over secure connections.",
    "Explain how rate limiting algorithms (Token Bucket vs Leaky Bucket) mitigate API DDoS exhaustion.",
    "Describe how cryptographic nonces prevent replay attacks in mutual authentication protocols.",
    "Explain how envelope encryption protects data at rest using Data Encryption Keys and KMS.",
    "Describe how security audits analyze Common Vulnerability Scoring System (CVSS) vectors.",
    "Explain how least privilege architecture minimizes blast radius in cloud identity policies.",
]
for i, p in enumerate(THEORY_TEMPLATES, 1):
    LARGE_SCALE_BENCHMARK.append({
        "id": f"THEORY_{i:02d}",
        "domain": "Defensive Cyber & Crypto",
        "expected": "theory",
        "prompt": p,
    })

# --- 33 AGENTIC ORCHESTRATION & STRUCTURED TOOL CALLING PROMPTS (agentic) ---
AGENTIC_TEMPLATES = [
    'Given tool definition query_sql_db(query: str), format a valid JSON tool call array to search for logs.',
    'Construct a strict JSON-RPC 2.0 request payload invoking cloud.deploy with nested CPU limit parameters.',
    'Explain how an agent handles an HTTP 429 rate limit using exponential backoff with jitter before retrying.',
    'Format a parallel tool execution block calling fetch_metrics(hostname="node1") and node2 simultaneously.',
    'Given a JSON schema requiring status: enum["PENDING", "ACTIVE"], demonstrate recovery from schema errors.',
    'Output a structured <|tool_call|> invocation matching tool definition read_file(path: str).',
    'Demonstrate how an agent orchestrates multiple consecutive tool response updates in a chat loop.',
    'Format a valid JSON tool call invoking bash(command="git status") with correct arguments.',
    'Given tool definition grep_search(path: str, query: str), generate a structured call to search for TODO.',
    'Demonstrate how to return a structured JSON response conforming to a strict Pydantic model schema.',
    'Format an agentic function call invoking patch_file(path: str, diff: str) safely.',
    'Explain how tool definition validation prevents hallucinated parameter names in LLM function calls.',
    'Generate a JSON-RPC response notification indicating successful background task completion.',
    'Construct a structured tool call array for list_dir(directory="/var/log") with recursive filtering.',
    'Demonstrate how an autonomous agent parses a multi-step execution plan into sequential tool calls.',
    'Format a valid JSON schema definition for a tool parameter accepting an array of string filters.',
    'Explain how an agent uses <|channel|>thought reasoning blocks before emitting a structured tool call.',
    'Construct a tool call payload invoking submit_flag(challenge_id=101, flag="HTB{test}") accurately.',
    'Demonstrate how to handle a tool response error message and generate an automatic self-correction turn.',
    'Format a parallel tool call invocation querying database status and checking disk usage concurrently.',
    'Given tool definition start_container(image: str), generate the exact JSON invocation payload.',
    'Explain how structured JSON grammar decoding guarantees syntactic validity for model tool outputs.',
    'Construct a tool response message block summarizing the output of an executed shell command.',
    'Format an agentic instruction that executes a multi-step search pipeline using grep_search tool.',
    'Demonstrate how an agent handles optional nullable parameters inside a JSON tool call definition.',
    'Construct a valid JSON payload invoking retrieve_ctf(target_ip="10.10.11.1") cleanly.',
    'Explain how tool call disambiguation resolves conflicting parameter types in overloaded schemas.',
    'Format a structured tool call array to update configuration settings via patch_file tool.',
    'Demonstrate how an agent validates JSON schema constraints before dispatching API requests.',
    'Construct an agentic workflow prompt that iterates over directory listings using list_dir tool.',
    'Format a JSON tool call invoking container_status(container_id="app-prod-1") with detail=true.',
    'Explain how structured tool calling enables deterministic integration with desktop C# / .NET clients.',
    'Generate a complete tool response block formatted as clean JSON for MatiApp client rendering.',
]
for i, p in enumerate(AGENTIC_TEMPLATES, 1):
    LARGE_SCALE_BENCHMARK.append({
        "id": f"AGENTIC_{i:02d}",
        "domain": "Agentic Tool Calling",
        "expected": "agentic",
        "prompt": p,
    })


def run_large_scale_benchmark():
    print("===================================================================")
    print("        MATI 12B MULTILORA-MOE 100-PROMPT BENCHMARK SUITE          ")
    print("===================================================================\n")

    engine = MatiMoEEngine()
    correct = 0
    total = len(LARGE_SCALE_BENCHMARK)

    domain_stats = {
        "Systems & Assembly": {"correct": 0, "total": 0},
        "Defensive Cyber & Crypto": {"correct": 0, "total": 0},
        "Agentic Tool Calling": {"correct": 0, "total": 0},
    }

    for task in LARGE_SCALE_BENCHMARK:
        turn = engine.generate_turn(task["prompt"])
        routing = turn["routing"]
        dominant = routing["dominant_expert"]

        passed = dominant == task["expected"]
        if passed:
            correct += 1
            domain_stats[task["domain"]]["correct"] += 1
        domain_stats[task["domain"]]["total"] += 1

    print("===================================================================")
    print("                100-PROMPT BENCHMARK ACCURACY SUMMARY              ")
    print("===================================================================")
    for dom, stats in domain_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f" * {dom:<28}: {stats['correct']:3d}/{stats['total']:3d} ({acc:5.1f}%)")
    print("-------------------------------------------------------------------")
    overall_acc = (correct / total) * 100
    print(f" OVERALL 100-TEST ROUTING ACCURACY : {correct:3d}/{total:3d} ({overall_acc:5.1f}%)")
    print("===================================================================\n")


def main():
    run_large_scale_benchmark()


if __name__ == "__main__":
    main()
