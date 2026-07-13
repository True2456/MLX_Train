#!/usr/bin/env python3
"""
Fuse MLX LoRA adapters into a Gemma model using strict=False to handle Gemma 4 unified/shim weights.
"""
import argparse
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_lm.utils import load_model, load_adapters, load_tokenizer, save

def fuse_and_save(base_model_path: str, adapter_path: str, save_path: str):
    print("=" * 60)
    print(f"Fusing Adapter: {adapter_path}")
    print(f"Base Model:     {base_model_path}")
    print(f"Destination:    {save_path}")
    print("=" * 60)

    base_path = Path(base_model_path)
    dest_path = Path(save_path)

    print("1/5 Loading base model weights (strict=False)...")
    model, config = load_model(base_path, lazy=False, strict=False)

    print("2/5 Loading LoRA adapter weights...")
    model = load_adapters(model, adapter_path)
    model.eval()

    print("3/5 Fusing LoRA weights into linear layers...")
    fused_linears = [
        (n, m.fuse(dequantize=False))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]
    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    print("4/5 Loading tokenizer...")
    tokenizer = load_tokenizer(base_path, {}, eos_token_ids=config.get("eos_token_id", None))

    print("5/5 Saving fused model to destination...")
    save(
        dest_path,
        base_path,
        model,
        tokenizer,
        config,
        donate_model=False,
    )
    print(f"\nSUCCESS! Saved fused model to: {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters with strict=False.")
    parser.add_argument("--base-model", default="models/gemma12b/base_gemma4_shim")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--save-path", required=True)
    args = parser.parse_args()

    fuse_and_save(args.base_model, args.adapter_path, args.save_path)
