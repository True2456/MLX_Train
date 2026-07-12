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
    ]

    THEORY_PATTERNS = [
        r"\b(CVE-\d{4}-\d+|\bCVSS\b|\bexploit\b|\bbuffer overflow\b|\bheap\b|\bsqli\b|\bxss\b)\b",
        r"\b(vulnerability|mitigation|privilege escalation|kernel|shellcode)\b",
    ]

    def __init__(self, default_theory_weight: float = 0.5):
        self.default_theory_weight = default_theory_weight

    def route_text(self, text: str) -> Tuple[float, float]:
        """Compute (w_theory, w_agentic) based on text content patterns."""
        agentic_score = sum(1 for p in self.AGENTIC_PATTERNS if re.search(p, text, re.IGNORECASE))
        theory_score = sum(1 for p in self.THEORY_PATTERNS if re.search(p, text, re.IGNORECASE))

        if agentic_score == 0 and theory_score == 0:
            w_t = self.default_theory_weight
            return (w_t, 1.0 - w_t)
        elif theory_score > 0 and agentic_score == 0:
            return (0.85, 0.15)
        elif agentic_score > 0 and theory_score == 0:
            return (0.15, 0.85)
        else:
            # Composite prompt where both domains are active: smooth proportional blend
            total = theory_score + agentic_score
            w_t = float(theory_score) / float(total)
            return (w_t, 1.0 - w_t)

    def route_batch(self, texts: list[str]) -> mx.array:
        """Return routing weights array of shape (batch_size, 2)."""
        weights = [self.route_text(t) for t in texts]
        return mx.array(weights, dtype=mx.float32)
