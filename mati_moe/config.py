"""Configuration definitions for 12B DualLoRA-MoE serving & routing."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass
class MultiExpertConfig:
    """Configuration for native 12B MultiLoRA-MoE runtime inside MLX/Mati.

    Matches the architecture specified in GEMMA12B_MOE_TRAINING_PLAN.md:
    - Shared base foundation on layers 0..7
    - Routed N-way MLP experts on layers 8..47
    - Auxiliary-loss-free bias routing (DeepSeek-V3 pattern)
    """

    d_model: int = 3840  # Gemma 4 12B hidden dimension
    num_experts: int = 3
    expert_names: Tuple[str, ...] = ("theory", "agentic", "asm_systems")

    # Layer routing range (inclusive)
    route_start_layer: int = 8
    route_end_layer: int = 47

    # Routing strategy: "soft_blend" (all experts weighted) or "top_1" or "top_k"
    routing_mode: str = "soft_blend"
    top_k: int = 2  # Active experts per token

    # DeepSeek-V3 style bias routing & regularization
    gamma: float = 0.001  # online bias update rate
    z_loss_coeff: float = 0.001
    warmup_steps: int = 150
    target_load: Optional[float] = None  # Expected equilibrium load per expert

    # Paths to expert LoRA adapter directories
    expert_adapter_paths: Optional[dict] = None

    # Mid-2026 Optimizations: MoE-Sieve & DR-LoRA
    sieve_threshold: float = 0.99
    layer_ranks: Optional[dict] = None

    def __post_init__(self):
        if self.num_experts < 2:
            raise ValueError(f"MultiExpertConfig requires num_experts>=2, got {self.num_experts}")
        if self.target_load is None:
            self.target_load = float(self.top_k) / float(self.num_experts)
        if self.route_start_layer < 0 or self.route_end_layer < self.route_start_layer:
            raise ValueError(f"Invalid layer range: [{self.route_start_layer}, {self.route_end_layer}]")
        if self.routing_mode not in ("soft_blend", "top_1", "top_k"):
            raise ValueError(f"Invalid routing_mode: {self.routing_mode}. Must be 'soft_blend', 'top_1', or 'top_k'")
        if not (0.0 < self.sieve_threshold <= 1.0):
            raise ValueError(f"Invalid sieve_threshold: {self.sieve_threshold}")
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")

    def save_json(self, path: Union[str, Path]) -> None:
        """Serialize configuration to JSON."""
        data = {
            "d_model": self.d_model,
            "num_experts": self.num_experts,
            "expert_names": list(self.expert_names),
            "route_start_layer": self.route_start_layer,
            "route_end_layer": self.route_end_layer,
            "routing_mode": self.routing_mode,
            "top_k": self.top_k,
            "gamma": self.gamma,
            "z_loss_coeff": self.z_loss_coeff,
            "warmup_steps": self.warmup_steps,
            "expert_adapter_paths": self.expert_adapter_paths,
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "MultiExpertConfig":
        """Load configuration from JSON."""
        data = json.loads(Path(path).read_text())
        if "expert_names" in data and isinstance(data["expert_names"], list):
            data["expert_names"] = tuple(data["expert_names"])
        return cls(**data)


class DualExpertConfig(MultiExpertConfig):
    """Backwards compatible 2-expert subclass of MultiExpertConfig."""

    def __init__(self, **kwargs):
        if "num_experts" not in kwargs:
            kwargs["num_experts"] = 2
        if "expert_names" not in kwargs:
            kwargs["expert_names"] = ("theory", "agentic")
        if "top_k" not in kwargs:
            kwargs["top_k"] = 1
        super().__init__(**kwargs)
