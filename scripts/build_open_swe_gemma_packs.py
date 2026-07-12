#!/usr/bin/env python3
"""Convert nvidia/Open-SWE-Traces (resolved OpenHands) → Gemma 4 native tool-call SFT packs.

Best-only policy:
  - config: openhands
  - split: minimax_m25  (thinking traces)
  - resolved == 1
  - map OpenHands tools → Mati-ish names in exact Gemma 4 token syntax
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "curated"

STR_DELIM = '<|"|>'


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:20]


def gemma_str(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    s = str(value)
    return f"{STR_DELIM}{s}{STR_DELIM}"


def format_tool_call(name: str, args: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(args.keys()):
        parts.append(f"{key}:{gemma_str(args[key])}")
    return f"<|tool_call>call:{name}{{{','.join(parts)}}}<tool_call|>"


def format_tool_response(name: str, value: str) -> str:
    # Truncate huge observations for training stability
    if len(value) > 12000:
        value = value[:12000] + "\n...[truncated]..."
    return f"<|tool_response>response:{name}{{value:{gemma_str(value)}}}<tool_response|>"


def format_thought(thought: str) -> str:
    thought = (thought or "").strip()
    if not thought:
        return ""
    return f"<|channel>thought\n{thought}\n<channel|>"


def parse_args_blob(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {"value": obj}
        except json.JSONDecodeError:
            return {"value": raw}
    return {"value": raw}


def map_str_replace_editor(args: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Map OpenHands str_replace_editor → Mati-like tools."""
    cmd = str(args.get("command") or args.get("cmd") or "").strip().lower()
    path = args.get("path") or args.get("file") or args.get("file_path") or "."
    path = str(path)

    if cmd in ("view", "open", "read"):
        # directory listing vs file — heuristic
        if path.endswith("/") or ("." not in Path(path).name and "view_range" not in args):
            # still often a file view; prefer read_file
            return "read_file", {"path": path}
        return "read_file", {"path": path}

    if cmd in ("create", "write"):
        content = args.get("file_text") or args.get("content") or args.get("new_str") or ""
        return "write_file", {"path": path, "content": str(content)}

    if cmd in ("str_replace", "replace", "edit"):
        old = args.get("old_str") or args.get("old") or ""
        new = args.get("new_str") or args.get("new") or ""
        # Mati patch_file often wants unified diff; keep explicit replace fields
        return "patch_file", {
            "path": path,
            "old_str": str(old),
            "new_str": str(new),
        }

    if cmd in ("insert", "append"):
        content = args.get("new_str") or args.get("content") or ""
        return "write_file", {"path": path, "content": str(content)}

    # Unknown editor command — keep as bash cat/sed fallback only if we have path
    if path:
        return "read_file", {"path": path}
    return None


def map_tool(name: str, args: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    n = (name or "").strip()
    if n in ("think", "noop"):
        return None  # folded into thought channel
    if n in ("execute_bash", "run_bash", "bash", "run", "terminal", "shell"):
        cmd = args.get("command") or args.get("cmd") or args.get("code") or ""
        return "bash", {"command": str(cmd)}
    if n in ("str_replace_editor", "editor", "edit_file"):
        return map_str_replace_editor(args)
    if n in ("finish", "submit", "done", "exit"):
        return None  # final answer path
    if n in ("read_file", "write_file", "patch_file", "grep", "list_dir", "bash"):
        return n, args
    # Pass through unknown tools with sanitized name
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", n)[:40] or "tool"
    return safe, args


def extract_user_task(trajectory: List[dict]) -> str:
    """Prefer the last non-system user message before tools start (issue statement)."""
    users = []
    for msg in trajectory:
        if (msg.get("role") or "").lower() != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # skip pure tool_response echoes
        if content.startswith("<|tool_response>"):
            continue
        users.append(content)
    if not users:
        return "Solve the software engineering issue in this repository."
    # Usually first user is system-ish env, last long one is the issue — pick longest
    return max(users, key=len)


def convert_trajectory(row: dict, max_turns: int, max_obs: int) -> Optional[dict]:
    traj = row.get("trajectory") or []
    if not traj:
        return None

    messages: List[dict] = []
    task = extract_user_task(traj)
    messages.append({"role": "user", "content": task})

    # Pair assistant tool_calls with following tool messages by order
    i = 0
    tool_turn_count = 0
    while i < len(traj):
        msg = traj[i]
        role = (msg.get("role") or "").lower()

        if role == "assistant":
            thought = (msg.get("think") or msg.get("reasoning_content") or "").strip()
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []

            # finish / final answer
            mapped_calls: List[Tuple[str, Dict[str, Any], str]] = []  # name, args, raw_id
            finish_only = False
            for tc in tool_calls:
                fn = tc.get("function") or {}
                raw_name = fn.get("name") or ""
                if raw_name in ("finish", "submit", "done"):
                    finish_only = True
                    # finish args may contain message
                    fargs = parse_args_blob(fn.get("arguments"))
                    final_msg = (
                        fargs.get("message")
                        or fargs.get("thought")
                        or fargs.get("summary")
                        or content
                        or "Task completed."
                    )
                    content = str(final_msg)
                    continue
                if raw_name == "think":
                    targs = parse_args_blob(fn.get("arguments"))
                    extra = str(targs.get("thought") or targs.get("content") or "")
                    if extra:
                        thought = (thought + "\n" + extra).strip() if thought else extra
                    continue
                mapped = map_tool(raw_name, parse_args_blob(fn.get("arguments")))
                if mapped:
                    mapped_calls.append((mapped[0], mapped[1], tc.get("id") or ""))

            if mapped_calls:
                body = format_thought(thought)
                for name, args, _ in mapped_calls:
                    body += format_tool_call(name, args)
                messages.append({"role": "assistant", "content": body})

                # Collect following tool responses (same count)
                responses = []
                j = i + 1
                while j < len(traj) and len(responses) < len(mapped_calls):
                    nxt = traj[j]
                    if (nxt.get("role") or "").lower() == "tool":
                        obs = (nxt.get("content") or "")[:max_obs]
                        responses.append(obs)
                        j += 1
                    elif (nxt.get("role") or "").lower() == "assistant":
                        break
                    else:
                        j += 1

                # Emit tool responses as user turn (Gemma SFT-friendly token string)
                if responses:
                    resp_body = ""
                    for k, obs in enumerate(responses):
                        name = mapped_calls[k][0] if k < len(mapped_calls) else "tool"
                        resp_body += format_tool_response(name, obs)
                    messages.append({"role": "user", "content": resp_body})
                    tool_turn_count += 1
                i = max(j, i + 1)
            else:
                # Final natural language / finish
                body = format_thought(thought)
                final = content or "Done."
                if body:
                    messages.append({"role": "assistant", "content": body + final})
                else:
                    messages.append({"role": "assistant", "content": final})
                i += 1

            if tool_turn_count >= max_turns:
                break
        else:
            i += 1

    # Must have at least one assistant tool turn or final answer
    if len(messages) < 2:
        return None
    if not any(m["role"] == "assistant" for m in messages):
        return None

    return {
        "messages": messages,
        "source": "open_swe_traces",
        "scaffold": "openhands",
        "teacher": "minimax_m25",
        "instance_id": row.get("instance_id"),
        "repo": row.get("repo"),
        "language": row.get("language"),
        "trajectory_id": row.get("trajectory_id"),
        "resolved": 1,
        "fingerprint": sha(json.dumps(messages, ensure_ascii=False)[:2000]),
    }


def also_emit_step_pairs(sample: dict, max_out: int) -> List[dict]:
    """Optional single-step SFT rows: each assistant tool turn as completion."""
    rows = []
    msgs = sample["messages"]
    for idx, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        if "<|tool_call>" not in m["content"] and "<|channel>thought" not in m["content"]:
            # final answer — still useful
            pass
        # Build a short prompt from prior context
        prior = msgs[:idx]
        # Keep last user only for compactness
        user_bits = [x["content"] for x in prior if x["role"] == "user"]
        if not user_bits:
            continue
        instruction = user_bits[0]
        if len(user_bits) > 1:
            instruction = (
                instruction[:2500]
                + "\n\n[...agent continued...]\n\n"
                + user_bits[-1][:8000]
            )
        output = m["content"]
        if len(output) > max_out:
            output = output[:max_out]
        rows.append(
            {
                "instruction": instruction[:12000],
                "input": "",
                "output": output,
                "source": "open_swe_traces_step",
                "instance_id": sample.get("instance_id"),
                "fingerprint": sha(instruction[:300] + output[:300]),
            }
        )
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(limit: int, max_turns: int, max_obs: int, seed: int, emit_steps: bool) -> dict:
    rng = random.Random(seed)
    print(
        f"Streaming nvidia/Open-SWE-Traces openhands/minimax_m25 "
        f"(resolved=1, limit={limit}, max_turns={max_turns})...",
        flush=True,
    )
    ds = load_dataset(
        "nvidia/Open-SWE-Traces",
        "openhands",
        split="minimax_m25",
        streaming=True,
    )

    kept: List[dict] = []
    seen = set()
    scanned = 0
    resolved_seen = 0
    skipped = Counter()

    for row in ds:
        scanned += 1
        if int(row.get("resolved") or 0) != 1:
            continue
        resolved_seen += 1
        sample = convert_trajectory(row, max_turns=max_turns, max_obs=max_obs)
        if not sample:
            skipped["convert_fail"] += 1
            continue
        # Prefer trajectories that actually used tools (agentic signal)
        if not any("<|tool_call>" in (m.get("content") or "") for m in sample["messages"]):
            skipped["no_tools"] += 1
            continue
        fp = sample["fingerprint"]
        if fp in seen:
            skipped["dup"] += 1
            continue
        seen.add(fp)
        kept.append(sample)
        if len(kept) % 100 == 0 or len(kept) >= limit:
            print(
                f"  scanned={scanned} resolved={resolved_seen} kept={len(kept)} skipped={dict(skipped)}",
                flush=True,
            )
        if len(kept) >= limit:
            break

    rng.shuffle(kept)

    # Best-only sizes: 12B gets a tighter 4k; 26B-A4B gets full curated set
    pack12 = kept[: min(4000, len(kept))]
    pack26 = kept[: min(limit, len(kept))]

    manifests = {}
    for name, pack, max_out in (
        ("agentic_gemma12b", pack12, 8000),
        ("agentic_gemma26b", pack26, 14000),
    ):
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        messages_path = out_dir / f"{name}_messages.jsonl"
        write_jsonl(messages_path, pack)

        step_rows: List[dict] = []
        if emit_steps:
            for s in pack:
                step_rows.extend(also_emit_step_pairs(s, max_out=max_out))
            # dedupe steps
            step_seen = set()
            uniq_steps = []
            for r in step_rows:
                if r["fingerprint"] in step_seen:
                    continue
                step_seen.add(r["fingerprint"])
                uniq_steps.append(r)
            step_rows = uniq_steps
            write_jsonl(out_dir / f"{name}_steps.jsonl", step_rows)

        langs = Counter(s.get("language") or "?" for s in pack)
        man = {
            "pack": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": "nvidia/Open-SWE-Traces",
            "filter": {
                "config": "openhands",
                "split": "minimax_m25",
                "resolved": 1,
                "limit": limit if name.endswith("26b") else min(4000, limit),
            },
            "trajectories": len(pack),
            "step_rows": len(step_rows),
            "languages": dict(langs.most_common()),
            "scanned": scanned,
            "resolved_seen": resolved_seen,
            "skipped": dict(skipped),
            "format": "gemma4_native_tool_tokens",
            "tool_mapping": {
                "execute_bash": "bash",
                "str_replace_editor.view": "read_file",
                "str_replace_editor.create": "write_file",
                "str_replace_editor.str_replace": "patch_file",
                "think": "channel_thought",
                "finish": "final_assistant_text",
            },
            "files": {
                "messages": str(messages_path),
                "steps": str(out_dir / f"{name}_steps.jsonl") if emit_steps else None,
            },
            "recommended_train": {
                "base_model": (
                    "mlx-community/gemma-4-12b-it-bf16"
                    if "12b" in name
                    else "mlx-community/gemma-4-26b-a4b-it-bf16"
                ),
                "max_seq_length": 4096 if "12b" in name else 8192,
                "notes": "Train on messages JSONL with Gemma chat template + tools, or steps JSONL via dingo instruction/output.",
            },
        }
        (out_dir / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        manifests[name] = man
        print(f"Wrote {name}: {len(pack)} trajectories, {len(step_rows)} step rows", flush=True)

    return manifests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=8000, help="Max resolved trajectories to keep")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--max-obs", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-steps", action="store_true")
    args = p.parse_args()
    build(
        limit=args.limit,
        max_turns=args.max_turns,
        max_obs=args.max_obs,
        seed=args.seed,
        emit_steps=not args.no_steps,
    )


if __name__ == "__main__":
    main()
