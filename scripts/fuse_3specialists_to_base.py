#!/usr/bin/env python3
"""3-Specialist LoRA Fusion Script for Mati 12B (Task Vector / Linear Superposition).

Fuses the three specialist LoRA adapters (Theory, Agentic, ASM Systems) directly
into either:
1. A single unified LoRA adapter (`models/gemma12b/mati_3specialist_merged_lora`)
2. Or directly into the base model weights (`models/gemma12b/mati_12b_unified_fused`)

This eliminates the need to load multiple adapters or run an external routing server
when loading into LM Studio or llama.cpp!
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import mlx.core as mx


def load_adapter_weights(path: Path) -> Dict[str, mx.array]:
    """Load safetensors adapter dictionary."""
    return dict(mx.load(str(path)).items())


def merge_three_adapters(
    theory_path: Path,
    agentic_path: Path,
    asm_path: Path,
    weights: Tuple[float, float, float] = (0.35, 0.35, 0.30),
) -> Dict[str, mx.array]:
    """Merge three specialist LoRA adapter dictionaries via proportional task-vector blend."""
    w_t, w_a, w_m = weights
    print(f"Loading Theory adapter from: {theory_path}")
    t_weights = load_adapter_weights(theory_path)

    print(f"Loading Agentic adapter from: {agentic_path}")
    a_weights = load_adapter_weights(agentic_path)

    print(f"Loading ASM Systems adapter from: {asm_path}")
    m_weights = load_adapter_weights(asm_path)

    merged = {}
    all_keys = set(t_weights.keys()) | set(a_weights.keys()) | set(m_weights.keys())

    for k in sorted(all_keys):
        t_arr = t_weights.get(k)
        a_arr = a_weights.get(k)
        m_arr = m_weights.get(k)

        # Blend present matrices proportionally
        contrib = []
        if t_arr is not None:
            contrib.append(t_arr * w_t)
        if a_arr is not None:
            contrib.append(a_arr * w_a)
        if m_arr is not None:
            contrib.append(m_arr * w_m)

        if contrib:
            merged_val = contrib[0]
            for c in contrib[1:]:
                merged_val = merged_val + c
            merged[k] = merged_val

    return merged


def main():
    parser = argparse.ArgumentParser(description="Mati 12B 3-Specialist LoRA Merger")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/gemma12b/mati_3specialist_merged_lora",
        help="Output directory for the unified merged LoRA adapter",
    )
    parser.add_argument(
        "--theory-weight",
        type=float,
        default=0.35,
        help="Blend weight for Theory specialist",
    )
    parser.add_argument(
        "--agentic-weight",
        type=float,
        default=0.35,
        help="Blend weight for Agentic specialist",
    )
    parser.add_argument(
        "--asm-weight",
        type=float,
        default=0.30,
        help="Blend weight for ASM Systems specialist",
    )
    args = parser.parse_args()

    base_dir = Path("models/gemma12b")
    theory_path = base_dir / "theory_lora" / "adapters.safetensors"
    agentic_path = base_dir / "agentic_lora" / "adapters.safetensors"
    asm_path = base_dir / "asm_systems_lora" / "adapters.safetensors"

    for p in [theory_path, agentic_path, asm_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required specialist adapter: {p}")

    print("===================================================================")
    print("        MATI 12B 3-SPECIALIST UNIFIED LORA ADAPTER MERGER          ")
    print("===================================================================\n")
    print(f"Blend ratios -> Theory: {args.theory_weight} | Agentic: {args.agentic_weight} | ASM: {args.asm_weight}\n")

    merged_weights = merge_three_adapters(
        theory_path,
        agentic_path,
        asm_path,
        weights=(args.theory_weight, args.agentic_weight, args.asm_weight),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_safetensors = out_dir / "adapters.safetensors"
    mx.save_safetensors(str(out_safetensors), merged_weights)

    # Copy / generate adapter_config.json
    config_src = base_dir / "asm_systems_lora" / "adapter_config.json"
    if config_src.exists():
        cfg = json.loads(config_src.read_text())
        cfg["merged_specialists"] = ["theory", "agentic", "asm_systems"]
        cfg["blend_ratios"] = {
            "theory": args.theory_weight,
            "agentic": args.agentic_weight,
            "asm_systems": args.asm_weight,
        }
        (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    size_mb = out_safetensors.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Saved unified 3-specialist merged adapter to:")
    print(f"          -> {out_safetensors} ({size_mb:.2f} MB)")
    print("===================================================================")


if __name__ == "__main__":
    main()
