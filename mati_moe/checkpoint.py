"""Checkpoint and bundle saving/loading utilities for mati_moe."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import mlx.core as mx
import mlx.utils as u
from .config import DualExpertConfig
from .router import DualExpertRouter


def save_mati_moe_bundle(
    output_dir: Union[str, Path],
    config: DualExpertConfig,
    router: DualExpertRouter,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a production mati_moe bundle containing config, router weights, and metadata.

    Args:
        output_dir: directory path to save bundle
        config: DualExpertConfig instance
        router: DualExpertRouter instance
        metadata: optional dictionary containing provenance, evaluation scores, etc.

    Returns:
        output_dir Path object
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save config.json
    config.save_json(out_path / "config.json")

    # 2. Save router safetensors (gate weights & expert_bias)
    weights = dict(u.tree_flatten(router.parameters()))
    mx.save_safetensors(str(out_path / "router.safetensors"), weights)

    # 3. Save bundle metadata if provided
    if metadata is not None:
        (out_path / "bundle_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    return out_path


def load_mati_moe_bundle(
    bundle_dir: Union[str, Path]
) -> Tuple[DualExpertConfig, DualExpertRouter, Dict[str, Any]]:
    """Load a production mati_moe bundle from disk.

    Args:
        bundle_dir: directory path containing config.json and router.safetensors

    Returns:
        tuple of (config, router, metadata)
    """
    b_path = Path(bundle_dir)
    config = DualExpertConfig.load_json(b_path / "config.json")

    router = DualExpertRouter(config)
    safetensors_file = b_path / "router.safetensors"
    if safetensors_file.exists():
        weights = mx.load(str(safetensors_file))
        router.load_weights(list(weights.items()))

    metadata: Dict[str, Any] = {}
    meta_file = b_path / "bundle_metadata.json"
    if meta_file.exists():
        metadata = json.loads(meta_file.read_text())

    return config, router, metadata
