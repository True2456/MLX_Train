#!/usr/bin/env python3
"""Split long agentic {prompt,completion} rows so they fit under max_seq_length.

The specialist mixes are already per-turn SFT rows, but late turns accumulate full
tool history and exceed the trainer gate (mlx_lm / forge skip when
prompt_tokens >= max_seq_length - 16). This script keeps the system+user prefix
and a trailing window of complete tool-exchange steps (split after
<tool_response|>), optionally truncating oversized tool payloads.

Does NOT start training. Output is a new JSONL + manifest beside the input.

Example:
  PYTHONPATH=~/Documents/gemma-forge/src \\
    python3 scripts/pack_agentic_sequences.py \\
      --input curated/specialists/gemma12b/agentic/train_steps.jsonl \\
      --output curated/specialists/gemma12b/agentic/train_steps_packed_8192.jsonl \\
      --max-seq-length 8192
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

GEN_SUFFIX = "<|turn>model\n<|channel>thought\n<channel|>"
GEN_SUFFIX_ALT = "<|turn>model\n"
USER_TURN_RE = re.compile(r"<\|turn>user\n.*?<turn\|>\n", re.DOTALL)
TOOL_RESP_VALUE_RE = re.compile(
    r"(<\|tool_response>response:\w+\{value:<\|\"\|>)(.*?)(<\|\"\|>\}<tool_response\|>)",
    re.DOTALL,
)
TRUNC_MARK = "\n...[truncated for max_seq]...\n"


def load_tokenizer(model_path: Path):
    """Load the same tokenizer the trainer uses, without loading model weights."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        return tok
    except Exception as exc:  # noqa: BLE001
        from tokenizers import Tokenizer

        print(f"transformers AutoTokenizer failed ({exc}); using tokenizer.json", flush=True)
        raw = Tokenizer.from_file(str(model_path / "tokenizer.json"))

        class _Tok:
            def encode(self, text: str, add_special_tokens: bool = True):  # noqa: ARG002
                return raw.encode(text).ids

        return _Tok()


def n_tokens(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text)
    # transformers may return BatchEncoding / list
    if hasattr(ids, "ids"):
        return len(ids.ids)
    return len(ids)


def split_prompt(prompt: str) -> tuple[str, list[str], str] | None:
    """Return (prefix, history_steps, gen_suffix) or None if structure unknown."""
    if prompt.endswith(GEN_SUFFIX):
        suffix = GEN_SUFFIX
        body = prompt[: -len(GEN_SUFFIX)]
    elif prompt.endswith(GEN_SUFFIX_ALT):
        suffix = GEN_SUFFIX_ALT
        body = prompt[: -len(GEN_SUFFIX_ALT)]
    else:
        idx = prompt.rfind(GEN_SUFFIX)
        if idx < 0:
            idx = prompt.rfind(GEN_SUFFIX_ALT)
            if idx < 0:
                return None
            suffix = prompt[idx:]
            body = prompt[:idx]
        else:
            suffix = prompt[idx:]
            body = prompt[:idx]

    m = USER_TURN_RE.search(body)
    if not m:
        return None
    prefix = body[: m.end()]
    hist = body[m.end() :]
    if not hist:
        return prefix, [], suffix

    raw_steps = re.split(r"(?<=<tool_response\|>)", hist)
    steps = [s for s in raw_steps if s]
    if not steps:
        steps = [hist]
    return prefix, steps, suffix


def ensure_model_turn(steps: list[str]) -> list[str]:
    """If the first kept step lost <|turn>model, re-attach it."""
    if not steps:
        return steps
    if steps[0].startswith("<|turn>model"):
        return steps
    out = list(steps)
    out[0] = "<|turn>model\n" + out[0]
    return out


def truncate_tool_values(text: str, max_chars: int) -> str:
    """Shrink oversized tool_response value payloads (keep structure)."""

    def _repl(match: re.Match[str]) -> str:
        head, val, tail = match.group(1), match.group(2), match.group(3)
        if len(val) <= max_chars:
            return match.group(0)
        keep = max(64, max_chars // 2)
        head_part = val[:keep]
        tail_part = val[-keep:] if keep < len(val) else ""
        return f"{head}{head_part}{TRUNC_MARK}{tail_part}{tail}"

    return TOOL_RESP_VALUE_RE.sub(_repl, text)


def fit_prompt(
    tokenizer,
    prompt: str,
    completion: str,
    *,
    max_seq_length: int,
    prompt_margin: int = 16,
) -> tuple[str, dict[str, Any]]:
    """Return a prompt that passes the forge trainer length gate + room for completion."""
    # Trainer skips when prompt_len >= max_seq_length - margin
    max_prompt = max_seq_length - prompt_margin - 1
    comp_tok = n_tokens(tokenizer, completion)
    # Keep completion trainable (avoid mlx truncating away the label)
    max_prompt = min(max_prompt, max_seq_length - comp_tok)
    max_prompt = max(256, max_prompt)

    meta: dict[str, Any] = {
        "action": "keep",
        "steps_total": 0,
        "steps_kept": 0,
        "truncated_payloads": False,
    }

    orig_tok = n_tokens(tokenizer, prompt)
    meta["tokens_in"] = orig_tok
    if orig_tok <= max_prompt:
        meta["tokens_out"] = orig_tok
        return prompt, meta

    parsed = split_prompt(prompt)
    if parsed is None:
        # Check for Expert 4 ASM/Systems canonical tags (<start_of_turn>user ... <start_of_turn>model\n)
        if prompt.startswith("<start_of_turn>user\n") and prompt.endswith("<start_of_turn>model\n"):
            meta["action"] = "asm_tail_preserve"
            header = "<start_of_turn>user\n[... truncated early context ...]\n"
            footer = "\n<end_of_turn>\n<start_of_turn>model\n"
            footer_idx = prompt.rfind("<end_of_turn>")
            if footer_idx != -1:
                body = prompt[len("<start_of_turn>user\n"):footer_idx]
                footer = prompt[footer_idx:]
            else:
                body = prompt[len("<start_of_turn>user\n"):]
            
            # Keep tail of body where assembly instructions / decompiled conclusions live
            ratio = max_prompt / max(orig_tok, 1)
            cut = max(512, int(len(body) * ratio * 0.92))
            tail_body = body[-cut:]
            new_prompt = header + tail_body + footer
            while n_tokens(tokenizer, new_prompt) > max_prompt and len(tail_body) > 256:
                tail_body = tail_body[len(tail_body) // 10:]
                new_prompt = header + tail_body + footer
            meta["tokens_out"] = n_tokens(tokenizer, new_prompt)
            return new_prompt, meta

        # Last resort: keep the tail of the prompt (rare non-native shape)
        meta["action"] = "hard_tail"
        ratio = max_prompt / max(orig_tok, 1)
        cut = max(512, int(len(prompt) * ratio * 0.95))
        new_prompt = prompt[-cut:]
        if not new_prompt.startswith("<bos>"):
            new_prompt = "<bos>" + new_prompt
        while n_tokens(tokenizer, new_prompt) > max_prompt and len(new_prompt) > 1024:
            new_prompt = "<bos>" + new_prompt[len("<bos>") + max(256, len(new_prompt) // 10) :]
        meta["tokens_out"] = n_tokens(tokenizer, new_prompt)
        return new_prompt, meta

    prefix, steps, suffix = parsed
    meta["steps_total"] = len(steps)
    prefix_tok = n_tokens(tokenizer, prefix)
    suffix_tok = n_tokens(tokenizer, suffix)
    overhead = prefix_tok + suffix_tok

    if overhead > max_prompt:
        # Prefix alone too big — truncate tool values inside prefix (tool decls are small;
        # user message may be huge) and/or hard-trim user content.
        meta["action"] = "shrink_prefix"
        user_m = USER_TURN_RE.search(prefix)
        if user_m:
            user_block = user_m.group(0)
            inner = user_block[len("<|turn>user\n") : -len("<turn|>\n")]
            budget_chars = max(512, (max_prompt - suffix_tok) * 3)
            if len(inner) > budget_chars:
                inner = inner[: budget_chars // 2] + TRUNC_MARK + inner[-(budget_chars // 2) :]
                prefix = prefix[: user_m.start()] + f"<|turn>user\n{inner}<turn|>\n"
                meta["truncated_payloads"] = True
        new_prompt = prefix + suffix
        while n_tokens(tokenizer, new_prompt) > max_prompt and len(new_prompt) > 1024:
            # peel from middle of prefix
            mid = len(prefix) // 2
            prefix = prefix[: mid // 2] + TRUNC_MARK + prefix[-(mid // 2) :]
            new_prompt = prefix + suffix
            meta["truncated_payloads"] = True
        meta["tokens_out"] = n_tokens(tokenizer, new_prompt)
        meta["steps_kept"] = 0
        return new_prompt, meta

    step_toks = [n_tokens(tokenizer, s) for s in steps]
    budget_hist = max_prompt - overhead

    # Take the longest suffix of steps that fits
    kept_from = len(steps)
    running = 0
    for i in range(len(steps) - 1, -1, -1):
        if running + step_toks[i] > budget_hist:
            break
        running += step_toks[i]
        kept_from = i

    kept = ensure_model_turn(steps[kept_from:])
    new_prompt = prefix + "".join(kept) + suffix
    out_tok = n_tokens(tokenizer, new_prompt)

    # Nothing fit, or still over: force-include last step with truncated payloads
    need_trunc = (not kept and bool(steps)) or out_tok > max_prompt
    if need_trunc:
        meta["truncated_payloads"] = True
        last = steps[-1]
        char_cap = max(256, budget_hist * 3)
        for _ in range(12):
            shrunk = truncate_tool_values(last, char_cap)
            kept = ensure_model_turn([shrunk] if shrunk else [])
            new_prompt = prefix + "".join(kept) + suffix
            out_tok = n_tokens(tokenizer, new_prompt)
            if out_tok <= max_prompt:
                break
            char_cap = max(128, char_cap // 2)
            last = shrunk
        # Absolute fallback: drop history entirely
        if out_tok > max_prompt:
            new_prompt = prefix + suffix
            out_tok = n_tokens(tokenizer, new_prompt)
            kept = []
        meta["action"] = "window_trunc"
        meta["steps_kept"] = len(kept)
        meta["tokens_out"] = out_tok
        return new_prompt, meta

    meta["action"] = "window" if kept_from > 0 else "keep"
    meta["steps_kept"] = len(kept)
    meta["tokens_out"] = out_tok
    return new_prompt, meta


def pack_file(
    *,
    input_path: Path,
    output_path: Path,
    tokenizer_path: Path,
    max_seq_length: int,
    limit: int | None = None,
    progress_every: int = 2000,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actions: Counter[str] = Counter()
    tokens_in: list[int] = []
    tokens_out: list[int] = []
    still_over = 0
    rows_in = 0
    rows_out = 0
    t0 = time.time()

    max_prompt_gate = max_seq_length - 16 - 1

    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows_in += 1
            if limit is not None and rows_in > limit:
                rows_in -= 1
                break

            row = json.loads(line)
            prompt = row.get("prompt") or ""
            completion = row.get("completion") or ""
            if not prompt or not completion:
                continue

            new_prompt, meta = fit_prompt(
                tokenizer,
                prompt,
                completion,
                max_seq_length=max_seq_length,
            )
            actions[meta["action"]] += 1
            tokens_in.append(int(meta["tokens_in"]))
            tokens_out.append(int(meta["tokens_out"]))

            out_row = dict(row)
            out_row["prompt"] = new_prompt
            if meta["action"] != "keep":
                out_row["pack_meta"] = {
                    "action": meta["action"],
                    "steps_total": meta.get("steps_total"),
                    "steps_kept": meta.get("steps_kept"),
                    "tokens_in": meta["tokens_in"],
                    "tokens_out": meta["tokens_out"],
                    "truncated_payloads": meta.get("truncated_payloads", False),
                    "max_seq_length": max_seq_length,
                }

            # Trainer-equivalent skip estimate after packing
            if meta["tokens_out"] > max_prompt_gate:
                still_over += 1
                # Drop rows that still cannot pass the gate
                continue

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            rows_out += 1

            if progress_every and rows_in % progress_every == 0:
                elapsed = time.time() - t0
                rate = rows_in / max(elapsed, 1e-6)
                print(
                    f"  … {rows_in:,} in / {rows_out:,} out "
                    f"({rate:.0f} rows/s, actions={dict(actions)})",
                    flush=True,
                )

    def _avg(xs: list[int]) -> float:
        return float(statistics.mean(xs)) if xs else 0.0

    def _pct(xs: list[int], p: float) -> int:
        if not xs:
            return 0
        ordered = sorted(xs)
        idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return int(ordered[idx])

    before_over = sum(1 for t in tokens_in if t > max_prompt_gate)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "tokenizer": str(tokenizer_path),
        "max_seq_length": max_seq_length,
        "prompt_gate": max_prompt_gate,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped_still_over": still_over,
        "before_over_gate": before_over,
        "before_over_pct": round(100.0 * before_over / max(rows_in, 1), 2),
        "after_over_gate": still_over,
        "after_over_pct": round(100.0 * still_over / max(rows_in, 1), 2),
        "actions": dict(actions),
        "tokens_in_avg": round(_avg(tokens_in), 1),
        "tokens_out_avg": round(_avg(tokens_out), 1),
        "tokens_in_p50": _pct(tokens_in, 50),
        "tokens_in_p95": _pct(tokens_in, 95),
        "tokens_out_p50": _pct(tokens_out, 50),
        "tokens_out_p95": _pct(tokens_out, 95),
        "elapsed_sec": round(time.time() - t0, 1),
        "notes": (
            "Split long tool histories on <tool_response|> boundaries; "
            "keep system+user prefix + recent steps. Rows still over the "
            "trainer prompt gate are dropped."
        ),
    }
    man_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.suffix == ".jsonl":
        man_path = output_path.with_name(output_path.stem + "_manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(man_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split/pack agentic SFT rows to fit max_seq_length"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "curated/specialists/gemma12b/agentic/train_steps.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <input_stem>_packed_<max_seq>.jsonl beside input",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "models/gemma12b/base_gemma4_shim",
        help="Gemma 4 shim or MLX model dir (tokenizer only)",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    tok_path = args.tokenizer.expanduser().resolve()
    if not tok_path.exists():
        print(f"ERROR: tokenizer path not found: {tok_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_packed_{args.max_seq_length}.jsonl"
        )
    else:
        output_path = output_path.expanduser().resolve()

    print(
        f"Packing {input_path}\n"
        f"  → {output_path}\n"
        f"  tokenizer={tok_path}\n"
        f"  max_seq_length={args.max_seq_length}",
        flush=True,
    )
    man = pack_file(
        input_path=input_path,
        output_path=output_path,
        tokenizer_path=tok_path,
        max_seq_length=args.max_seq_length,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps({k: man[k] for k in man if k != "notes"}, indent=2), flush=True)
    print(f"Wrote manifest → {man['manifest_path']}", flush=True)


if __name__ == "__main__":
    main()
