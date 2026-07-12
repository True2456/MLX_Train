#!/usr/bin/env python3
"""Build theory/agentic specialist packs for Gemma 4 12B and 26B-A4B fusion.

Layout (under curated/specialists/):
  gemma12b/{theory,agentic}/
  gemma26b/{theory,agentic}/

Agentic mix (best for your fuse plan):
  S  personal gold (Done HTB + filtered desktop + mati-eval) — oversampled
  A  flawless_native_pack (already gemma_native)
  B  Open-SWE resolved messages — capped, converted to prompt/completion

Theory packs are linked from existing theory_gemma{12,26}b builds.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOJO = Path.home() / "Documents/GojoCode"
FORGE = Path.home() / "Documents/gemma-forge"
RED = Path.home() / "Documents/red team"
OUT = ROOT / "curated" / "specialists"

# gemma-forge on path
sys.path.insert(0, str(FORGE / "src"))
from forge.tools.normalize_trajectory import normalize_trajectory  # noqa: E402
from forge.train.trainer import rows_from_trajectory, write_training_jsonl  # noqa: E402
from forge.wire.gemma_native import (  # noqa: E402
    DEFAULT_NATIVE_RULES,
    NativeMessage,
    ToolResponse,
    build_workspace_tools,
    format_tool_response_block,
    render_prompt,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:20]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def personal_eligible(row: dict, *, min_turns: int, require_verified: bool) -> tuple[bool, str]:
    turns = row.get("turns") or []
    if len(turns) < min_turns:
        return False, "too_few_turns"
    if require_verified and not (row.get("verified") or row.get("sandbox_success")):
        return False, "not_verified"
    if row.get("sandbox_success") is False and not row.get("verified"):
        return False, "sandbox_failed"
    # Must have at least one real tool action
    toolish = 0
    for t in turns:
        at = (t.get("action") or {}).get("type") or t.get("action_type") or "none"
        if at and at != "none":
            toolish += 1
    if toolish < 1:
        return False, "no_tools"
    return True, "ok"


def collect_done_htb() -> list[dict]:
    out = []
    for name in ("Challenge 5 (Done)", "Challenge 11 (Done)", "bowmaster (Done)"):
        path = RED / name / ".mati/logs/trajectories_verified.jsonl"
        for row in load_jsonl(path):
            row = dict(row)
            row["_origin"] = f"red_team/{name}"
            row["_tier"] = "S_done_htb"
            out.append(row)
    return out


def collect_desktop_filtered() -> list[dict]:
    path = GOJO / "data/archives/mati-desktop/trajectories_verified.jsonl"
    out = []
    for row in load_jsonl(path):
        agent = (row.get("agent_name") or "mati_desktop").lower()
        turns = len(row.get("turns") or [])
        # Main agent: keep all verified
        if agent in ("mati_desktop", "mati", "main", ""):
            row = dict(row)
            row["_origin"] = "mati_desktop_main"
            row["_tier"] = "S_desktop_main"
            out.append(row)
            continue
        # Subagents: keep only multi-turn specialists (drop 1-turn prover spam)
        if turns >= 5 and agent not in ("prover",):
            row = dict(row)
            row["_origin"] = f"mati_desktop_{agent}"
            row["_tier"] = "S_desktop_sub"
            out.append(row)
        elif turns >= 8 and agent == "prover":
            # rare long prover runs only
            row = dict(row)
            row["_origin"] = "mati_desktop_prover_long"
            row["_tier"] = "S_desktop_sub"
            out.append(row)
    return out


def collect_mati_eval() -> list[dict]:
    path = GOJO / "data/archives/mati-eval/trajectories_verified.jsonl"
    out = []
    for row in load_jsonl(path):
        row = dict(row)
        row["_origin"] = "mati_eval"
        row["_tier"] = "S_mati_eval"
        out.append(row)
    return out


def normalize_personal(rows: list[dict]) -> list[dict]:
    kept = []
    seen = set()
    skipped = Counter()
    for row in rows:
        ok, reason = personal_eligible(row, min_turns=3, require_verified=False)
        if not ok:
            # Done HTB / eval: allow if sandbox_success true even if verify flag missing
            if row.get("_tier", "").startswith("S_") and row.get("sandbox_success") is not False:
                if len(row.get("turns") or []) >= 3:
                    ok = True
                else:
                    skipped[reason] += 1
                    continue
            else:
                skipped[reason] += 1
                continue
        norm = normalize_trajectory(
            row,
            source=row.get("_origin") or "personal",
            tier=row.get("_tier") or "S",
        )
        if not norm:
            skipped["norm_fail"] += 1
            continue
        # Looser than forge flawless: do not require final action == none
        fp = sha(norm["instruction"][:400] + json.dumps(norm["turns"], ensure_ascii=False)[:800])
        if fp in seen:
            skipped["dup"] += 1
            continue
        seen.add(fp)
        norm["fingerprint"] = fp
        norm["_origin"] = row.get("_origin")
        norm["_tier"] = row.get("_tier")
        kept.append(norm)
    print(f"  personal normalize: kept={len(kept)} skipped={dict(skipped)}", flush=True)
    return kept


def oversample(rows: list[dict], factor: int) -> list[dict]:
    if factor <= 1:
        return list(rows)
    out = []
    for i in range(factor):
        for r in rows:
            c = dict(r)
            c["curated_oversample"] = i
            out.append(c)
    return out


def open_swe_to_steps(messages_path: Path, *, limit_traj: int, max_turns: int, seed: int) -> list[dict]:
    """Convert Open-SWE messages JSONL → forge-style {prompt,completion} steps."""
    tools = build_workspace_tools()
    rng = random.Random(seed)
    trajs = load_jsonl(messages_path)
    rng.shuffle(trajs)
    trajs = trajs[:limit_traj]
    steps: list[dict] = []
    for sample in trajs:
        msgs = sample.get("messages") or []
        if not msgs:
            continue
        history: list[NativeMessage] = [
            NativeMessage("system", DEFAULT_NATIVE_RULES),
        ]
        turn_n = 0
        for msg in msgs:
            role = (msg.get("role") or "").lower()
            content = msg.get("content") or ""
            if role == "user":
                # tool responses already embedded as <|tool_response> in user content
                if content.startswith("<|tool_response>"):
                    # Attach as tool_responses on previous assistant if possible
                    history.append(NativeMessage("user", content))
                else:
                    history.append(NativeMessage("user", content))
            elif role == "assistant":
                prompt = render_prompt(
                    history,
                    tools,
                    enable_thinking=False,
                    add_generation_prompt=True,
                )
                completion = content
                useful = bool(completion.strip()) and (
                    "<|tool_call>" in completion
                    or "<|channel>thought" in completion
                    or turn_n == 0
                )
                if useful:
                    steps.append(
                        {
                            "prompt": prompt,
                            "completion": completion,
                            "source": "open_swe",
                            "instance_id": sample.get("instance_id"),
                        }
                    )
                    turn_n += 1
                history.append(NativeMessage("assistant", content))
                if turn_n >= max_turns:
                    break
    return steps


def link_or_copy_theory(size: str) -> Path:
    src = ROOT / "curated" / f"theory_gemma{size}"
    dst = OUT / f"gemma{size}" / "theory"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    # Prefer symlink to avoid duplicating ~100MB+
    dst.symlink_to(src.resolve())
    return dst


def build_agentic(
    size: str,
    *,
    personal: list[dict],
    personal_factor: int,
    open_swe_limit: int,
    open_swe_max_turns: int,
    seed: int,
) -> dict:
    out_dir = OUT / f"gemma{size}" / "agentic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # S: personal → steps
    personal_os = oversample(personal, personal_factor)
    personal_steps_path = out_dir / "personal_gold_steps.jsonl"
    tools = build_workspace_tools()
    personal_stats = write_training_jsonl(
        personal_os,
        str(personal_steps_path),
        tools=tools,
        require_success=False,  # already filtered
    )
    write_jsonl(out_dir / "personal_gold_trajectories.jsonl", personal)

    # A: flawless
    flawless_src = FORGE / "data/curated/flawless_native_pack.jsonl"
    flawless_rows = load_jsonl(flawless_src)

    # B: Open-SWE (from existing curated messages)
    swe_src = ROOT / "curated" / f"agentic_gemma{size}" / f"agentic_gemma{size}_messages.jsonl"
    if not swe_src.exists():
        # fallback to 26b/12b sibling
        alt = "26b" if size == "12b" else "12b"
        swe_src = ROOT / "curated" / f"agentic_gemma{alt}" / f"agentic_gemma{alt}_messages.jsonl"
    swe_steps = open_swe_to_steps(
        swe_src,
        limit_traj=open_swe_limit,
        max_turns=open_swe_max_turns,
        seed=seed,
    )

    # Mix: personal steps + flawless + open-swe (personal already oversampled on disk)
    personal_steps = load_jsonl(personal_steps_path)
    mixed: list[dict] = []
    for r in personal_steps:
        mixed.append({**r, "mix_tier": "S_personal"})
    for r in flawless_rows:
        mixed.append({**r, "mix_tier": "A_flawless"})
    for r in swe_steps:
        mixed.append({**r, "mix_tier": "B_open_swe"})

    rng = random.Random(seed)
    rng.shuffle(mixed)
    train_path = out_dir / "train_steps.jsonl"
    write_jsonl(train_path, mixed)

    # Also keep source pointers
    if flawless_src.exists():
        link = out_dir / "flawless_native_pack.jsonl"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(flawless_src.resolve())

    tier_counts = Counter(r["mix_tier"] for r in mixed)
    man = {
        "pack": f"gemma{size}_agentic_specialist",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "agentic_expert",
        "base_model_hint": (
            "mlx-community/gemma-4-12b-it-bf16"
            if size == "12b"
            else "mlx-community/gemma-4-26b-a4b-it-bf16"
        ),
        "wire_format": "gemma_native_prompt_completion",
        "train_file": str(train_path),
        "rows": len(mixed),
        "tiers": dict(tier_counts),
        "personal_trajectories": len(personal),
        "personal_oversample_factor": personal_factor,
        "personal_stats": personal_stats,
        "open_swe_source": str(swe_src),
        "open_swe_traj_cap": open_swe_limit,
        "recommended_train": {
            "max_seq_length": 4096 if size == "12b" else 8192,
            "lora_rank": 16 if size == "12b" else 32,
            "notes": (
                "Train ONLY this pack for the agentic expert. "
                "Do not mix theory JSONL into this run. "
                "After both experts exist, fuse via branch-train MoE / router — not naive weight average."
            ),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote gemma{size} agentic: {len(mixed)} steps "
        f"(S={tier_counts['S_personal']} A={tier_counts['A_flawless']} B={tier_counts['B_open_swe']})",
        flush=True,
    )
    return man


def main() -> None:
    print("Collecting personal gold…", flush=True)
    raw = collect_done_htb() + collect_desktop_filtered() + collect_mati_eval()
    print(
        f"  raw candidates: done_htb={sum(1 for r in raw if r.get('_tier')=='S_done_htb')} "
        f"desktop={sum(1 for r in raw if str(r.get('_tier','')).startswith('S_desktop'))} "
        f"eval={sum(1 for r in raw if r.get('_tier')=='S_mati_eval')}",
        flush=True,
    )
    personal = normalize_personal(raw)

    # Theory links
    for size in ("12b", "26b"):
        link_or_copy_theory(size)
        print(f"Linked theory → specialists/gemma{size}/theory", flush=True)

    # Agentic mixes — 12B tighter, 26B fuller
    m12 = build_agentic(
        "12b",
        personal=personal,
        personal_factor=40,  # tiny gold → strong weight
        open_swe_limit=2500,
        open_swe_max_turns=24,
        seed=42,
    )
    m26 = build_agentic(
        "26b",
        personal=personal,
        personal_factor=50,
        open_swe_limit=5000,
        open_swe_max_turns=32,
        seed=42,
    )

    recipe = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "goal": "Train two same-size specialists (theory + agentic), then fuse",
        "pairs": [
            {
                "name": "gemma12b_fuse",
                "theory": "curated/specialists/gemma12b/theory",
                "agentic": "curated/specialists/gemma12b/agentic/train_steps.jsonl",
                "base": "gemma-4-12b-it",
            },
            {
                "name": "gemma26b_fuse",
                "theory": "curated/specialists/gemma26b/theory",
                "agentic": "curated/specialists/gemma26b/agentic/train_steps.jsonl",
                "base": "gemma-4-26b-a4b-it",
            },
        ],
        "agentic_manifests": {"12b": m12, "26b": m26},
        "fusion_notes": [
            "Train theory and agentic experts separately from the SAME base checkpoint.",
            "Do not train one model on mixed theory+tools — that collapses the specialist signal.",
            "Prefer branch-train + learned router over franken-MoE of unrelated checkpoints.",
            "Personal HTB gold is oversampled so it is not drowned by Open-SWE volume.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Specialists at {OUT}", flush=True)


if __name__ == "__main__":
    main()
