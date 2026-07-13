"""Model patching utilities to attach DualExpertRouter and RoutedMLP to Gemma 4 12B."""

from typing import Dict, List
import mlx.core as mx
import mlx.nn as nn
from .config import DualExpertConfig
from .mlp import DualLoraMLP, RoutedMLP
from .router import DualExpertRouter


def patch_gemma4_moe(
    model: nn.Module,
    router: DualExpertRouter,
    config: DualExpertConfig,
    expert_theory_mlps: Dict[int, nn.Module],
    expert_agentic_mlps: Dict[int, nn.Module],
) -> List[RoutedMLP]:
    """Patch layers in model.layers[config.route_start_layer : config.route_end_layer + 1].

    Replaces layer.mlp on routed layers with a RoutedMLP instance that handles
    both dynamic gating and dual expert blending transparently.

    Args:
        model: loaded MLX Gemma 4 12B model
        router: shared DualExpertRouter instance
        config: DualExpertConfig specifying start/end routed layers
        expert_theory_mlps: mapping from layer index to theory specialist MLP
        expert_agentic_mlps: mapping from layer index to agentic specialist MLP

    Returns:
        patched_modules: list of RoutedMLP instances attached to the model
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model must have .model.layers attribute (standard MLX LLM structure)")

    layers = model.model.layers
    patched_modules: List[RoutedMLP] = []

    for idx, layer in enumerate(layers):
        if config.route_start_layer <= idx <= config.route_end_layer:
            if idx not in expert_theory_mlps or idx not in expert_agentic_mlps:
                continue

            dual_mlp = DualLoraMLP(
                expert_theory=expert_theory_mlps[idx],
                expert_agentic=expert_agentic_mlps[idx],
            )
            routed_mlp = RoutedMLP(router=router, dual_mlp=dual_mlp)
            routed_mlp._original_mlp = getattr(layer, "mlp", None)

            # Replace layer.mlp with drop-in RoutedMLP
            layer.mlp = routed_mlp
            patched_modules.append(routed_mlp)

    return patched_modules


def unpatch_gemma4_moe(model: nn.Module) -> int:
    """Restore original .mlp modules on any layers patched with RoutedMLP.

    Args:
        model: MLX Gemma 4 model previously patched by patch_gemma4_moe

    Returns:
        num_unpatched: number of layers restored to their original MLP
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return 0

    num_unpatched = 0
    for layer in model.model.layers:
        if isinstance(getattr(layer, "mlp", None), RoutedMLP):
            original = getattr(layer.mlp, "_original_mlp", None)
            if original is not None:
                layer.mlp = original
            num_unpatched += 1

    return num_unpatched
