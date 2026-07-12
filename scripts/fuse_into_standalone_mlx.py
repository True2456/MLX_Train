#!/usr/bin/env python3
"""True Multi-Specialist Standalone MLX Shard Merger for Mati 12B.

Mathematically computes the full linear delta (B^T @ A^T * scale) for each specialist
adapter independently before combining them:
  Delta_W = w_theory * (B_theory^T @ A_theory^T * scale)
          + w_agentic * (B_agentic^T @ A_agentic^T * scale)
          + w_asm * (B_asm^T @ A_asm^T * scale)
  W_fused = W_base + Delta_W

This avoids SVD cross-term sign cancellation and ensures that cybersecurity, tool-calling,
and assembly decompilation capabilities are fully embedded into the standalone model.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple
import mlx.core as mx


def load_adapter(path: Path) -> Dict[str, mx.array]:
    if not path.exists():
        raise FileNotFoundError(f"Adapter file not found: {path}")
    return dict(mx.load(str(path)).items())


def fuse_shard_multi_specialist(
    shard_path: Path,
    out_path: Path,
    adapters: list[Tuple[Dict[str, mx.array], float]],
    scale: float = 10.0,
):
    """Load a safetensors shard, compute sum of specialist deltas, and save out."""
    shard_weights = dict(mx.load(str(shard_path)).items())
    fused_count = 0

    for name, w in list(shard_weights.items()):
        if not name.endswith(".weight"):
            continue
        prefix = name[:-7]
        lora_a_name = prefix + ".lora_a"
        lora_b_name = prefix + ".lora_b"

        delta_total = None

        for adapter_dict, blend_weight in adapters:
            if lora_a_name in adapter_dict and lora_b_name in adapter_dict:
                a = adapter_dict[lora_a_name]
                b = adapter_dict[lora_b_name]
                # True LoRA linear shift: (B^T @ A^T) * scale
                delta_i = mx.matmul(b.T, a.T) * (scale * blend_weight)
                if delta_total is None:
                    delta_total = delta_i
                else:
                    delta_total = delta_total + delta_i

        if delta_total is not None:
            shard_weights[name] = (w + delta_total.astype(w.dtype)).astype(w.dtype)
            fused_count += 1

    mx.save_safetensors(str(out_path), shard_weights)
    return fused_count


def main():
    parser = argparse.ArgumentParser(description="Mati 12B True Multi-Specialist Shard Merger")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/Users/true/.lmstudio/models/mlx-community/gemma-4-12B-it-bf16",
        help="Path to base Gemma 4 12B MLX model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/true/.lmstudio/models/mlx-community/mati-12b-unified-fused-bf16",
        help="Output directory for the standalone fused MLX model",
    )
    parser.add_argument("--theory-weight", type=float, default=0.40, help="Blend weight for Theory / Cyber specialist")
    parser.add_argument("--agentic-weight", type=float, default=0.30, help="Blend weight for Agentic specialist")
    parser.add_argument("--asm-weight", type=float, default=0.30, help="Blend weight for ASM Systems specialist")
    parser.add_argument("--scale", type=float, default=10.0, help="LoRA scaling factor")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base model dir not found: {base_dir}")

    print("===================================================================")
    print("      MATI 12B TRUE SPECIALIST LINEAR FUSION MERGER ENGINE         ")
    print("===================================================================\n")
    print(f"Base Model:    {base_dir}")
    print(f"Output Folder: {output_dir}")
    print(
        f"Blend Ratios:  Theory/Cyber={args.theory_weight} | Agentic={args.agentic_weight} | ASM={args.asm_weight}\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy config and non-safetensor metadata files
    print("1. Copying model configurations and tokenizer assets...")
    for f in base_dir.iterdir():
        if not f.name.endswith(".safetensors") and f.is_file():
            shutil.copy2(f, output_dir / f.name)
            print(f"   -> Copied {f.name}")

    # 2. Load all 3 specialist adapters separately
    print("\n2. Loading individual specialist adapters...")
    models_dir = Path("models/gemma12b")
    theory_adapter = load_adapter(models_dir / "theory_lora" / "adapters.safetensors")
    agentic_adapter = load_adapter(models_dir / "agentic_lora" / "adapters.safetensors")
    asm_adapter = load_adapter(models_dir / "asm_systems_lora" / "adapters.safetensors")

    adapters = [
        (theory_adapter, args.theory_weight),
        (agentic_adapter, args.agentic_weight),
        (asm_adapter, args.asm_weight),
    ]

    # 3. Fuse shard by shard
    shards = sorted(base_dir.glob("model-*.safetensors"))
    if not shards:
        single = base_dir / "model.safetensors"
        if single.exists():
            shards = [single]
        else:
            raise RuntimeError(f"No .safetensors shards found in {base_dir}")

    print(f"\n3. Computing true linear deltas and fusing into {len(shards)} model shards...")
    total_fused = 0
    for i, shard in enumerate(shards, 1):
        out_shard = output_dir / shard.name
        print(f"   [{i}/{len(shards)}] Processing {shard.name}...", end="", flush=True)
        count = fuse_shard_multi_specialist(shard, out_shard, adapters, scale=args.scale)
        total_fused += count
        size_mb = out_shard.stat().st_size / (1024 * 1024)
        print(f" Fused {count} tensors -> Saved ({size_mb:.1f} MB)")

    print(f"\n[SUCCESS] Standalone MLX model created at: {output_dir}")
    print(f"          Total fused weight tensors: {total_fused}")
    print("          True linear delta W = w1*(B1*A1) + w2*(B2*A2) + w3*(B3*A3) applied!")
    print("===================================================================")


if __name__ == "__main__":
    main()
