"""DualLoraMLP and RoutedMLP wrappers blending specialist MLPs."""

from typing import List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from .router import DualExpertRouter


class MultiLoraMLP(nn.Module):
    """Wraps N specialist MLPs (e.g. theory, agentic, asm_systems) and mixes their outputs via router weights.

    Supports:
    - N materialized expert MLPs (or LoRA-adapted MLPs)
    - MoE-Sieve Fast-Path Gating (July 2026 Breakthrough) for any expert
    """

    def __init__(self, experts: List[nn.Module], expert_names: Optional[List[str]] = None):
        super().__init__()
        self.experts = experts
        self.num_experts = len(experts)
        self.expert_names = expert_names or [f"expert_{i}" for i in range(self.num_experts)]
        self.last_sieved_expert: Optional[str] = None

    def __call__(self, x: mx.array, weights: mx.array, sieve_threshold: float = 1.0) -> mx.array:
        """Forward pass blending N expert outputs with optional MoE-Sieve fast-path gating."""
        # MoE-Sieve Fast-Path Gating (July 2026 Breakthrough)
        if sieve_threshold < 1.0:
            for i in range(self.num_experts):
                w_i = weights[..., i : i + 1]
                if mx.all(w_i >= sieve_threshold).item():
                    self.last_sieved_expert = self.expert_names[i]
                    return self.experts[i](x)

        self.last_sieved_expert = "none"
        blended = mx.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w_i = weights[..., i : i + 1]
            blended = blended + w_i * expert(x)
        return blended


class DualLoraMLP(MultiLoraMLP):
    """Wraps two specialist MLPs (theory & agentic) and mixes their outputs via router weights.

    Backwards compatible wrapper around MultiLoraMLP.
    """

    def __init__(self, expert_theory: nn.Module, expert_agentic: nn.Module):
        super().__init__(
            experts=[expert_theory, expert_agentic],
            expert_names=["theory", "agentic"],
        )
        self.expert_theory = expert_theory
        self.expert_agentic = expert_agentic


class RoutedMLP(nn.Module):
    """Drop-in replacement for layer.mlp in standard MLX transformer blocks.

    Executes:
    1. Router gating on input hidden state x -> (weights, logits, z_loss)
    2. DualLoraMLP blending -> blended output
    3. Caches last_weights, last_logits, and last_z_loss for telemetry & CI gates
    """

    def __init__(self, router: DualExpertRouter, dual_mlp: DualLoraMLP):
        super().__init__()
        self.router = router
        self.dual_mlp = dual_mlp
        self.last_weights: Optional[mx.array] = None
        self.last_logits: Optional[mx.array] = None
        self.last_z_loss: Optional[mx.array] = None

    def __call__(self, x: mx.array, update_bias: bool = False) -> mx.array:
        """Forward pass compatible with standard layer.mlp(x) calls."""
        weights, logits, z_loss = self.router(x, update_bias=update_bias)
        self.last_weights = weights
        self.last_logits = logits
        self.last_z_loss = z_loss

        return self.dual_mlp(x, weights, sieve_threshold=self.router.config.sieve_threshold)
