#!/usr/bin/env python3
"""Shard-by-Shard Standalone MLX Fusion Script for Mati 12B MultiLoRA-MoE.

Directly fuses the unified specialist LoRA weights into the base Gemma 4 12B
safetensors shards, producing a standalone MLX model folder that can be opened
directly in LM Studio or served anywhere with zero extra dependencies.
"""

import argparse
import shutil
from pathlib import Path
import mlx.core as mx


def fuse_shard(
    shard_path: Path,
    out_path: Path,
    adapter_weights: dict,
    scale: float = 10.0,
):
    """Load a safetensors shard, fuse matching LoRA weights, and save out."""
    shard_weights = dict(mx.load(str(shard_path)).items())
    fused_count = 0

    for name, w in list(shard_weights.items()):
        if not name.endswith(".weight"):
            continue
        prefix = name[:-7]
        lora_a_name = prefix + ".lora_a"
        lora_b_name = prefix + ".lora_b"

        if lora_a_name in adapter_weights and lora_b_name in adapter_weights:
            a = adapter_weights[lora_a_name]
            b = adapter_weights[lora_b_name]
            # W shape is (out_features, in_features)
            # a shape is (in_features, rank), b shape is (rank, out_features)
            delta = mx.matmul(b.T, a.T) * scale
            shard_weights[name] = (w + delta.astype(w.dtype)).astype(w.dtype)
            fused_count += 1

    mx.save_safetensors(str(out_path), shard_weights)
    return fused_count


def main():
    parser = argparse.ArgumentParser(description="Mati 12B Standalone MLX Shard Merger")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/Users/true/.lmstudio/models/mlx-community/gemma-4-12B-it-bf16",
        help="Path to base Gemma 4 12B MLX model directory",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="models/gemma12b/mati_3specialist_merged_lora/adapters.safetensors",
        help="Path to unified merged adapter safetensors file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/gemma12b/mati_12b_unified_fused",
        help="Output directory for the standalone fused MLX model",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="LoRA scaling factor (default 10.0 from adapter_config)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    adapter_file = Path(args.adapter_file)
    output_dir = Path(args.output_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base model dir not found: {base_dir}")
    if not adapter_file.exists():
        raise FileNotFoundError(f"Merged adapter file not found: {adapter_file}")

    print("===================================================================")
    print("      MATI 12B SHARD-BY-SHARD STANDALONE MLX FUSION ENGINE         ")
    print("===================================================================\n")
    print(f"Base Model:     {base_dir}")
    print(f"Merged Adapter: {adapter_file}")
    print(f"Output Folder:  {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy config and non-safetensor metadata files
    print("1. Copying model configurations and tokenizer assets...")
    for f in base_dir.iterdir():
        if not f.name.endswith(".safetensors") and f.is_file():
            shutil.copy2(f, output_dir / f.name)
            print(f"   -> Copied {f.name}")

    # 2. Load merged adapter weights
    print(f"\n2. Loading unified specialist adapter ({adapter_file.stat().st_size / (1024*1024):.1f} MB)...")
    adapter_weights = dict(mx.load(str(adapter_file)).items())

    # 3. Fuse shard by shard
    shards = sorted(base_dir.glob("model-*.safetensors"))
    if not shards:
        # Check single model.safetensors
        single = base_dir / "model.safetensors"
        if single.exists():
            shards = [single]
        else:
            raise RuntimeError(f"No .safetensors shards found in {base_dir}")

    print(f"\n3. Fusing weights into {len(shards)} model shards...")
    total_fused = 0
    for i, shard in enumerate(shards, 1):
        out_shard = output_dir / shard.name
        print(f"   [{i}/{len(shards)}] Processing {shard.name}...", end="", flush=True)
        count = fuse_shard(shard, out_shard, adapter_weights, scale=args.scale)
        total_fused += count
        size_mb = out_shard.stat().st_size / (1024 * 1024)
        print(f" Fused {count} tensors -> Saved ({size_mb:.1f} MB)")

    print(f"\n[SUCCESS] Standalone MLX model created at: {output_dir}")
    print(f"          Total fused weight tensors: {total_fused}")
    print("          You can now open this folder directly in LM Studio!")
    print("===================================================================")


if __name__ == "__main__":
    main()
