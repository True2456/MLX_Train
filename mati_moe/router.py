"""Linear bias-based loss-free router for 12B DualLoRA-MoE (DeepSeek V3/V4 pattern)."""

from typing import Tuple
import mlx.core as mx
import mlx.nn as nn
from .config import DualExpertConfig


class DualExpertRouter(nn.Module):
    """Router module applied to pre-FFN hidden states at each routed layer.

    Features:
    - Linear projection from d_model to num_experts (2)
    - Online scalar bias per expert updated without backpropagation (DeepSeek V3/V4 pattern)
    - Soft/weighted blend routing or top-1 gating
    - Logits z-loss computation
    """

    def __init__(self, config: DualExpertConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)

        # Auxiliary-loss-free bias tracking vector b_e
        self.expert_bias = mx.zeros((config.num_experts,))

    def update_bias(self, weights: mx.array) -> None:
        """Online bias update rule (DeepSeek-V3 pattern, gamma = 0.001).

        Decreases bias for overloaded experts and increases bias for underloaded experts.
        """
        # Mean load across batch & sequence tokens
        # weights shape: (..., num_experts)
        flat_weights = mx.reshape(weights, (-1, self.num_experts))
        mean_load = mx.mean(flat_weights, axis=0)
        target_load = getattr(self.config, "target_load", float(getattr(self.config, "top_k", 1)) / self.num_experts)

        # b_e -= gamma * sign(mean_load - target_load) or proportional delta
        load_diff = mean_load - target_load
        self.expert_bias = self.expert_bias - self.config.gamma * load_diff
        mx.eval(self.expert_bias)

    def compute_z_loss(self, logits: mx.array) -> mx.array:
        """Compute logsumexp z-loss to prevent logit drift."""
        # z_loss = coeff * mean(logsumexp(logits)^2)
        lse = mx.logsumexp(logits, axis=-1)
        return self.config.z_loss_coeff * mx.mean(lse * lse)

    def __call__(self, hidden_states: mx.array, update_bias: bool = False) -> Tuple[mx.array, mx.array, mx.array]:
        """Route hidden states to expert weights.

        Args:
            hidden_states: input tensor of shape (..., d_model)
            update_bias: whether to perform online bias update on this step

        Returns:
            weights: routing weights of shape (..., num_experts)
            logits: raw gate logits of shape (..., num_experts)
            z_loss: scalar z-loss value
        """
        logits = self.gate(hidden_states)
        biased_logits = logits + self.expert_bias

        if self.config.routing_mode == "soft_blend":
            weights = mx.softmax(biased_logits, axis=-1)
        else:
            top_k = min(getattr(self.config, "top_k", 1), self.num_experts)
            probs = mx.softmax(biased_logits, axis=-1)
            sorted_probs = mx.sort(probs, axis=-1)
            thresh = sorted_probs[..., -1:] if top_k == 1 else sorted_probs[..., -top_k : -top_k + 1]
            mask = probs >= thresh
            masked_probs = mx.where(mask, probs, 0.0)
            denom = mx.sum(masked_probs, axis=-1, keepdims=True)
            weights = mx.where(denom > 0.0, masked_probs / denom, probs)

        z_loss = self.compute_z_loss(logits)

        if update_bias:
            self.update_bias(weights)

        return weights, logits, z_loss
