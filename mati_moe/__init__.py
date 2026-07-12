"""mati_moe: Native 12B DualLoRA-MoE serving & routing backend for MLX/Mati."""

from .checkpoint import load_mati_moe_bundle, save_mati_moe_bundle
from .config import DualExpertConfig
from .heuristic_router import HeuristicRouter
from .mlp import DualLoraMLP, RoutedMLP
from .patching import patch_gemma4_moe, unpatch_gemma4_moe
from .router import DualExpertRouter

__all__ = [
    "DualExpertConfig",
    "DualExpertRouter",
    "HeuristicRouter",
    "DualLoraMLP",
    "RoutedMLP",
    "patch_gemma4_moe",
    "unpatch_gemma4_moe",
    "save_mati_moe_bundle",
    "load_mati_moe_bundle",
]
