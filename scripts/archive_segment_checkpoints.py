#!/usr/bin/env python3
"""Archive forge/mlx numbered LoRA checkpoints with a global segment prefix.

mlx_lm / gemma-forge write per-segment names like ``0000050_adapters.safetensors``.
On resume the Iter counter resets, so a new segment would silently overwrite those
files. Run this **before** starting a new train segment to rename them to
``seg{N}_0000050_adapters.safetensors``.

Leaves live resume weights untouched:
  - adapters.safetensors
  - adapter_config.json

Idempotent: already-prefixed ``seg*_…`` files are skipped.

Examples:
  python3 scripts/archive_segment_checkpoints.py models/gemma12b/theory_lora
  python3 scripts/archive_segment_checkpoints.py models/gemma12b/agentic_lora --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Live segment (unprefixed) numbered adapters from mlx_lm steps_per_save.
LIVE_CKPT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")
# Optional sibling artifacts mlx may write alongside a numbered save.
LIVE_RELATED_RE = re.compile(r"^(\d+)_(adapters\.safetensors|adapter_config\.json)$")
# Already archived.
SEG_PREFIX_RE = re.compile(r"^seg(\d+)_")


def next_segment_index(adapter_dir: Path) -> int:
    highest = 0
    for p in adapter_dir.iterdir():
        if not p.is_file():
            continue
        m = SEG_PREFIX_RE.match(p.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def collect_live_checkpoints(adapter_dir: Path) -> list[Path]:
    """Unprefixed numbered checkpoint files only (not seg*, not adapters.safetensors)."""
    found: dict[str, Path] = {}
    for p in adapter_dir.iterdir():
        if not p.is_file():
            continue
        if SEG_PREFIX_RE.match(p.name):
            continue
        if p.name in {"adapters.safetensors", "adapter_config.json", "DONE"}:
            continue
        if LIVE_RELATED_RE.match(p.name) or LIVE_CKPT_RE.match(p.name):
            found[p.name] = p
    # Stable order by numeric iter then name
    def sort_key(path: Path) -> tuple[int, str]:
        m = re.match(r"^(\d+)_", path.name)
        return (int(m.group(1)) if m else 0, path.name)

    return sorted(found.values(), key=sort_key)


def archive_segment(
    adapter_dir: Path,
    *,
    dry_run: bool = False,
    segment: int | None = None,
) -> list[tuple[Path, Path]]:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"LoRA output dir not found: {adapter_dir}")

    live = collect_live_checkpoints(adapter_dir)
    if not live:
        return []

    seg = segment if segment is not None else next_segment_index(adapter_dir)
    if seg < 1:
        raise ValueError(f"segment index must be >= 1, got {seg}")

    renames: list[tuple[Path, Path]] = []
    for src in live:
        dst = adapter_dir / f"seg{seg}_{src.name}"
        if dst.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing archived checkpoint: {dst}"
            )
        renames.append((src, dst))

    if not dry_run:
        for src, dst in renames:
            src.rename(dst)

    return renames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rename unprefixed forge numbered LoRA checkpoints with seg{N}_ "
            "before a resume segment, leaving adapters.safetensors untouched."
        )
    )
    parser.add_argument(
        "adapter_dir",
        type=Path,
        help="LoRA output directory (e.g. models/gemma12b/theory_lora)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without changing files",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Force segment index N (default: next unused seg*)",
    )
    args = parser.parse_args(argv)

    adapter_dir = args.adapter_dir.expanduser().resolve()
    try:
        renames = archive_segment(
            adapter_dir, dry_run=args.dry_run, segment=args.segment
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not renames:
        print(f"No unprefixed numbered checkpoints in {adapter_dir} — nothing to archive.")
        return 0

    seg = int(re.match(r"^seg(\d+)_", renames[0][1].name).group(1))  # type: ignore[union-attr]
    mode = "DRY-RUN" if args.dry_run else "ARCHIVED"
    print(f"{mode} segment seg{seg} in {adapter_dir}")
    print(f"  {len(renames)} file(s):")
    for src, dst in renames:
        print(f"    {src.name}  →  {dst.name}")
    print("  Left untouched: adapters.safetensors, adapter_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
