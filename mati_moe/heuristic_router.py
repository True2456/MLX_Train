"""Heuristic router for zero-parameter v1 validation gate."""

import re
from typing import Tuple
import mlx.core as mx


class HeuristicRouter:
    """Zero-parameter heuristic router for validating the forward wrapper.

    Detects tool schemas, tool keywords, or agentic tokens vs. theory keywords
    to emit synthetic routing weights (w_theory, w_agentic).
    """

    AGENTIC_PATTERNS = [
        r"<\|tool_call\|>",
        r"<\|tool_response\|>",
        r"<\|channel\|>thought",
        r"\b(read_file|write_file|patch_file|bash|grep_search|list_dir)\b",
        r"```(json|bash)",
        # HTB MCP / agent harness cues (route to agentic specialist)
        r"\b(list_ctf_events|retrieve_ctf|start_container|stop_container|submit_flag|container_status)\b",
        r"\b(htb|hack\s*the\s*box|mcp)\b",
        r"\b(call:?\w+|tool_call|function.?call|tool definition|json tool call|json-rpc|tool response|json schema|exponential backoff|schema)\b",
    ]

    ASM_PATTERNS = [
        r"\b(pushq|popq|movq|subq|addq|leaq|callq|retq|rbp|rsp|rdi|rsi|x86_64|arm64|riscv)\b",
        r"\b(decompile|assembly|asm|abi|calling convention|struct padding|sizeof\(void\*\)|elf|elf64|ehdr|binary header|msr|rdmsr|cpuid|got|plt|mach-o|tss|gdb|strace|ptrace|core dump|unwinding|c frame)\b",
        r"```(asm|c)",
    ]

    THEORY_PATTERNS = [
        r"\b(CVE-\d{4}-\d+|\bCVSS\b|\bexploit\b|\bbuffer overflow\b|\bheap\b|\bsqli\b|\bxss\b)\b",
        r"\b(vulnerability|mitigation|privilege escalation|kernel|shellcode)\b",
    ]

    def __init__(self, default_theory_weight: float = 0.5):
        self.default_theory_weight = default_theory_weight

    def route_text(self, text: str) -> Tuple[float, float]:
        """Compute (w_theory, w_agentic) based on text content patterns (backwards compatible 2-way)."""
        w_t, w_a, _ = self.route_3way(text)
        total = w_t + w_a
        if total > 0:
            return (w_t / total, w_a / total)
        return (self.default_theory_weight, 1.0 - self.default_theory_weight)

    def route_3way(self, text: str) -> Tuple[float, float, float]:
        """Compute (w_theory, w_agentic, w_asm) 3-way specialist routing weights."""
        agentic_score = sum(1 for p in self.AGENTIC_PATTERNS if re.search(p, text, re.IGNORECASE))
        asm_score = sum(1 for p in self.ASM_PATTERNS if re.search(p, text, re.IGNORECASE))
        theory_score = sum(1 for p in self.THEORY_PATTERNS if re.search(p, text, re.IGNORECASE))

        total = agentic_score + asm_score + theory_score
        if total == 0:
            return (0.34, 0.33, 0.33)

        w_t = float(theory_score) / total
        w_a = float(agentic_score) / total
        w_m = float(asm_score) / total
        return (w_t, w_a, w_m)

    def route_batch(self, texts: list[str]) -> mx.array:
        """Return routing weights array of shape (batch_size, 2)."""
        weights = [self.route_text(t) for t in texts]
        return mx.array(weights, dtype=mx.float32)
