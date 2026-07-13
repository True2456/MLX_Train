"""Production Serving Engine for Mati 12B MultiLoRA-MoE (N=3 Specialists).

Orchestrates:
1. Dynamic 3-way expert routing across Theory, Agentic, and ASM Systems
2. MoE-Sieve fast-path gating telemetry
3. Seamless integration with MLX Gemma 4 transformer blocks
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from .config import MultiExpertConfig
from .heuristic_router import HeuristicRouter
from .mlp import MultiLoraMLP, RoutedMLP
from .patching import patch_gemma4_moe, unpatch_gemma4_moe
from .router import DualExpertRouter


class MatiMoEEngine:
    """Unified Serving & Routing Controller for Mati 12B MultiLoRA-MoE."""

    def __init__(
        self,
        config: Optional[MultiExpertConfig] = None,
        model: Optional[nn.Module] = None,
    ):
        self.config = config or MultiExpertConfig(
            num_experts=3,
            expert_names=("theory", "agentic", "asm_systems"),
        )
        self.heuristic_router = HeuristicRouter()
        self.router = DualExpertRouter(self.config)
        self.model = model
        self.patched_layers: List[RoutedMLP] = []

    def route_prompt(self, prompt: str) -> Dict[str, Any]:
        """Compute routing telemetry across all 3 specialist experts."""
        w_t, w_a, w_m = self.heuristic_router.route_3way(prompt)

        sieve_expert = "none"
        if w_t >= self.config.sieve_threshold:
            sieve_expert = "theory"
        elif w_a >= self.config.sieve_threshold:
            sieve_expert = "agentic"
        elif w_m >= self.config.sieve_threshold:
            sieve_expert = "asm_systems"

        return {
            "weights": {
                "theory": round(w_t, 4),
                "agentic": round(w_a, 4),
                "asm_systems": round(w_m, 4),
            },
            "sieved_expert": sieve_expert,
            "fast_path_triggered": sieve_expert != "none",
            "dominant_expert": max(
                [("theory", w_t), ("agentic", w_a), ("asm_systems", w_m)],
                key=lambda item: item[1],
            )[0],
        }

    def verify_checkpoints_ready(self, base_dir: Union[str, Path] = "models/gemma12b") -> Dict[str, Dict[str, Any]]:
        """Verify on-disk availability of all 3 selected production specialist adapters."""
        b_path = Path(base_dir)
        checkpoints = {
            "theory": {
                "selected_iter": 11350,
                "loss": None,
                "path": b_path / "theory_lora" / "adapters.safetensors",
            },
            "agentic": {
                "selected_iter": "seg8_200",
                "loss": "converged multi-turn",
                "path": b_path / "agentic_lora" / "adapters.safetensors",
            },
            "asm_systems": {
                "selected_iter": 2500,
                "loss": 0.043,
                "path": b_path / "asm_systems_lora" / "adapters.safetensors",
            },
        }

        results = {}
        for name, spec in checkpoints.items():
            exists = spec["path"].exists()
            size_mb = spec["path"].stat().st_size / (1024 * 1024) if exists else 0.0
            results[name] = {
                "ready": exists,
                "path": str(spec["path"]),
                "size_mb": round(size_mb, 2),
                "selected_iter": spec["selected_iter"],
                "loss": spec["loss"],
            }
        return results

    def generate_turn(self, prompt: str) -> Dict[str, Any]:
        """Execute or simulate a routing and generation turn through the MultiLoRA-MoE stack."""
        start_t = time.perf_counter()
        telemetry = self.route_prompt(prompt)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "prompt": prompt,
            "routing": telemetry,
            "latency_ms": round(elapsed_ms, 3),
            "status": "SUCCESS",
        }
