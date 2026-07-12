"""DualLoraMLP and RoutedMLP wrappers blending specialist MLPs."""

from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from .router import DualExpertRouter


class DualLoraMLP(nn.Module):
    """Wraps two specialist MLPs (theory & agentic) and mixes their outputs via router weights.

    Can wrap:
    - Two materialized expert MLPs (e.g. fused theory MLP vs fused agentic MLP)
    - Or two LoRA-adapted MLPs sharing the same base weights
    """

    def __init__(self, expert_theory: nn.Module, expert_agentic: nn.Module):
        super().__init__()
        self.expert_theory = expert_theory
        self.expert_agentic = expert_agentic
        self.last_sieved_expert: Optional[str] = None

    def __call__(self, x: mx.array, weights: mx.array, sieve_threshold: float = 1.0) -> mx.array:
        """Forward pass blending expert outputs with optional MoE-Sieve fast-path gating.

        Args:
            x: input hidden states of shape (..., d_model)
            weights: routing weights tensor of shape (..., 2) where
                     weights[..., 0] is theory weight and
                     weights[..., 1] is agentic weight
            sieve_threshold: threshold above which un-routed expert is skipped (MoE-Sieve 2026)

        Returns:
            blended hidden states of shape (..., d_model)
        """
        w_theory = weights[..., 0:1]
        w_agentic = weights[..., 1:2]

        # MoE-Sieve Fast-Path Gating (July 2026 Breakthrough)
        # Re-normalizes active expert weight to 1.0 to prevent residual magnitude decay
        if sieve_threshold < 1.0:
            if mx.all(w_theory >= sieve_threshold).item():
                self.last_sieved_expert = "theory"
                return self.expert_theory(x)
            if mx.all(w_agentic >= sieve_threshold).item():
                self.last_sieved_expert = "agentic"
                return self.expert_agentic(x)

        self.last_sieved_expert = "none"
        out_theory = self.expert_theory(x)
        out_agentic = self.expert_agentic(x)

        blended = w_theory * out_theory + w_agentic * out_agentic
        return blended


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
