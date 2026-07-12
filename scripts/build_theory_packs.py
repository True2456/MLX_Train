#!/usr/bin/env python3
"""Build Gemma theory SFT packs (12B + 26B) from Mati_Train Tier-1 sources.

Output format matches dingo MLX trainer non-agent rows:
  {"instruction": "...", "input": "", "output": "...", "source": "..."}

Also writes a messages-format twin for mlx-lm / generic chat SFT.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "theory"
OUT_DIR = ROOT / "curated"

# Security-ish CLI titles to keep from the huge man-page dump (substring match).
CLI_KEEP = re.compile(
    r"(nmap|tcpdump|wireshark|tshark|iptables|nft|sshd|openssl|gpg|curl|wget|"
    r"nc\.|netcat|socat|strace|lsof|gdb|objdump|readelf|binwalk|hashcat|john|"
    r"hydra|sqlmap|nikto|gobuster|ffuf|masscan|arp|route|ip\b|ifconfig|"
    r"dig|nslookup|whois|traceroute|ping|chmod|chown|sudo|su\b|passwd|"
    r"auditd|journalctl|systemctl|docker|kubectl|aws|gcloud|az\b)",
    re.I,
)

NON_CYBER_GENERAL = re.compile(
    r"(旅游|攻略|美食|购物|酒店|签证|机票|香港|东京|巴黎|recipe|travel|tourism|hotel)",
    re.I,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:20]


def trunc(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 20]
    # prefer paragraph boundary
    for sep in ("\n\n", "\n", ". "):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.6:
            cut = cut[: idx + len(sep)].rstrip()
            break
    return cut + "\n\n[...truncated for training length...]"


def messages_to_io(messages: Any) -> Optional[Tuple[str, str]]:
    """Extract last user + last assistant from Primus-style messages."""
    if messages is None:
        return None
    # pyarrow may give list of dicts or numpy array of dicts
    try:
        msgs = list(messages)
    except TypeError:
        return None
    user = ""
    assistant = ""
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role in ("user", "human"):
            user = content
        elif role in ("assistant", "model"):
            assistant = content
    if user and assistant:
        return user, assistant
    return None


def load_parquet(path: Path):
    return pq.read_table(path).to_pandas()


def add_row(
    rows: List[dict],
    seen: set,
    *,
    instruction: str,
    output: str,
    source: str,
    subset: str,
    max_out: int,
) -> None:
    instruction = (instruction or "").strip()
    output = trunc((output or "").strip(), max_out)
    if len(instruction) < 8 or len(output) < 40:
        return
    fp = sha(instruction[:400] + "||" + output[:400])
    if fp in seen:
        return
    seen.add(fp)
    rows.append(
        {
            "instruction": instruction,
            "input": "",
            "output": output,
            "source": source,
            "subset": subset,
            "fingerprint": fp,
        }
    )


def ingest_primus_instruct(rows: List[dict], seen: set, max_out: int) -> Counter:
    counts: Counter = Counter()
    data = THEORY / "primus/Primus-Instruct/data"
    for path in sorted(data.glob("*.parquet")):
        subset = path.stem
        df = load_parquet(path)
        for _, r in df.iterrows():
            if subset == "general":
                prompt = str(r.get("prompt") or "")
                if NON_CYBER_GENERAL.search(prompt):
                    counts["primus_instruct_general_skip"] += 1
                    continue
            pair = messages_to_io(r.get("messages"))
            if not pair:
                counts["primus_instruct_skip"] += 1
                continue
            user, assistant = pair
            before = len(rows)
            add_row(
                rows,
                seen,
                instruction=user,
                output=assistant,
                source="primus_instruct",
                subset=subset,
                max_out=max_out,
            )
            if len(rows) > before:
                counts[f"primus_instruct:{subset}"] += 1
    return counts


def ingest_primus_reasoning(rows: List[dict], seen: set, max_out: int, prefer: str) -> Counter:
    """prefer: 'r1' | 'o1' | 'both' — avoid near-duplicate CVE tasks."""
    counts: Counter = Counter()
    data = THEORY / "primus/Primus-Reasoning/data"
    files = []
    if prefer in ("r1", "both"):
        files.append(data / "ctibench_deepseek-r1.parquet")
    if prefer in ("o1", "both"):
        files.append(data / "ctibench_o1.parquet")
    for path in files:
        if not path.exists():
            continue
        df = load_parquet(path)
        for _, r in df.iterrows():
            pair = messages_to_io(r.get("messages"))
            if not pair:
                # fall back to prompt-only if messages incomplete
                prompt = str(r.get("prompt") or "").strip()
                if not prompt:
                    counts["primus_reasoning_skip"] += 1
                    continue
                # without assistant we can't train — skip
                counts["primus_reasoning_skip"] += 1
                continue
            user, assistant = pair
            before = len(rows)
            add_row(
                rows,
                seen,
                instruction=user,
                output=assistant,
                source="primus_reasoning",
                subset=path.stem,
                max_out=max_out,
            )
            if len(rows) > before:
                counts[f"primus_reasoning:{path.stem}"] += 1
    return counts


def ingest_redsage(rows: List[dict], seen: set, max_out: int, include_cli: bool, skills_cap: Optional[int], rng: random.Random) -> Counter:
    counts: Counter = Counter()
    base = THEORY / "redsage-seed/RedSage-Seed"
    configs = [
        ("cybersecurity_knowledge_frameworks", "framework", "Explain this security framework / ATT&CK technique in clear practical terms:\n\nTitle: {title}"),
        ("cybersecurity_knowledge_generals", "general", "Explain this cybersecurity concept for an advanced practitioner:\n\nTopic: {title}"),
        ("cybersecurity_skills", "skills", "Walk through this offensive/CTF case study. Explain the attack path, key primitives, and lessons:\n\nTitle: {title}"),
        ("cybersecurity_tools_kali", "kali", "Explain this Kali/security tooling topic and when to use it:\n\nTitle: {title}"),
    ]
    if include_cli:
        configs.append(
            (
                "cybersecurity_tools_cli",
                "cli",
                "Summarize this command/tool reference for a security engineer (purpose, key flags, caveats):\n\nTitle: {title}",
            )
        )

    for folder, subset, tmpl in configs:
        path = base / folder / "train-00000-of-00001.parquet"
        if not path.exists():
            continue
        df = load_parquet(path)
        records = list(df.to_dict(orient="records"))
        if subset == "skills" and skills_cap and len(records) > skills_cap:
            records = rng.sample(records, skills_cap)
        if subset == "cli":
            filtered = []
            for r in records:
                title = str(r.get("title") or r.get("id") or "")
                if CLI_KEEP.search(title):
                    filtered.append(r)
            records = filtered

        for r in records:
            title = str(r.get("title") or r.get("id") or "untitled").strip()
            body = str(r.get("refined_content") or r.get("content") or "").strip()
            if not body:
                continue
            instruction = tmpl.format(title=title)
            before = len(rows)
            add_row(
                rows,
                seen,
                instruction=instruction,
                output=body,
                source="redsage",
                subset=subset,
                max_out=max_out,
            )
            if len(rows) > before:
                counts[f"redsage:{subset}"] += 1
    return counts


def ingest_ctftime(rows: List[dict], seen: set, max_out: int, sample_n: int, rng: random.Random) -> Counter:
    counts: Counter = Counter()
    path = THEORY / "substitutes-for-cybersecurity-1m/ctf/CTFtime/data/train-00000-of-00001.parquet"
    if not path.exists():
        return counts
    df = load_parquet(path)
    records = list(df.to_dict(orient="records"))
    # Prefer mid-length writeups
    scored = []
    for r in records:
        text = str(r.get("text_chunk") or "").strip()
        if len(text) < 200:
            continue
        scored.append(text)
    if sample_n and len(scored) > sample_n:
        scored = rng.sample(scored, sample_n)
    for text in scored:
        first = text.split("\n", 1)[0][:160].strip()
        body = trunc(text, max_out)
        instruction = (
            "You are a cybersecurity mentor. Rewrite the following CTF writeup as a clear teaching explanation with:\n"
            "1) challenge overview\n2) recon findings\n3) vulnerability / primitive\n"
            "4) exploit steps\n5) key lessons.\n\n"
            f"Header: {first}\n\nWriteup:\n{body}"
        )
        # Supervise on a structured teaching rewrite target = the writeup itself
        # (model learns to preserve technical detail under a teaching prompt).
        output = body
        before = len(rows)
        add_row(
            rows,
            seen,
            instruction=instruction,
            output=output,
            source="ctftime",
            subset="writeup",
            max_out=max_out,
        )
        if len(rows) > before:
            counts["ctftime:writeup"] += 1
    return counts


def ingest_kev(rows: List[dict], seen: set, max_out: int, sample_n: int, rng: random.Random) -> Counter:
    counts: Counter = Counter()
    path = THEORY / "substitutes-for-cybersecurity-1m/vuln-curated/known_exploited_vulnerabilities.json"
    if not path.exists():
        return counts
    data = json.loads(path.read_text(encoding="utf-8"))
    vulns = data.get("vulnerabilities") or []
    if sample_n and len(vulns) > sample_n:
        vulns = rng.sample(vulns, sample_n)
    for v in vulns:
        cve = v.get("cveID") or ""
        name = v.get("vulnerabilityName") or ""
        vendor = v.get("vendorProject") or ""
        product = v.get("product") or ""
        desc = v.get("shortDescription") or ""
        action = v.get("requiredAction") or ""
        due = v.get("dueDate") or ""
        if not cve or not desc:
            continue
        instruction = (
            f"Explain CISA KEV entry {cve} ({name}). Cover what is affected, why it matters, "
            f"and the required defensive action."
        )
        output = (
            f"CVE: {cve}\n"
            f"Name: {name}\n"
            f"Vendor/Product: {vendor} / {product}\n"
            f"Description: {desc}\n"
            f"Required action: {action}\n"
            f"Due date: {due}\n"
            f"Notes: This vulnerability is on CISA's Known Exploited Vulnerabilities catalog, "
            f"meaning active exploitation has been observed. Prioritize patching or the mandated mitigation."
        )
        before = len(rows)
        add_row(
            rows,
            seen,
            instruction=instruction,
            output=output,
            source="cisa_kev",
            subset="kev",
            max_out=max_out,
        )
        if len(rows) > before:
            counts["cisa_kev"] += 1
    return counts


def to_messages_row(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ],
        "source": row["source"],
        "subset": row["subset"],
        "fingerprint": row["fingerprint"],
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_pack(name: str, cfg: dict, seed: int = 42) -> dict:
    rng = random.Random(seed)
    rows: List[dict] = []
    seen: set = set()
    stats: Counter = Counter()

    stats.update(ingest_primus_instruct(rows, seen, cfg["max_out"]))
    stats.update(ingest_primus_reasoning(rows, seen, cfg["max_out"], cfg["reasoning"]))
    stats.update(
        ingest_redsage(
            rows,
            seen,
            cfg["max_out"],
            include_cli=cfg["include_cli"],
            skills_cap=cfg.get("skills_cap"),
            rng=rng,
        )
    )
    stats.update(ingest_ctftime(rows, seen, cfg["max_out"], cfg["ctftime_n"], rng))
    stats.update(ingest_kev(rows, seen, cfg["max_out"], cfg["kev_n"], rng))

    rng.shuffle(rows)

    out_root = OUT_DIR / name
    out_root.mkdir(parents=True, exist_ok=True)
    alpaca_path = out_root / f"{name}_train.jsonl"
    messages_path = out_root / f"{name}_messages.jsonl"
    write_jsonl(alpaca_path, rows)
    write_jsonl(messages_path, (to_messages_row(r) for r in rows))

    by_source = Counter(r["source"] for r in rows)
    by_subset = Counter(f"{r['source']}:{r['subset']}" for r in rows)
    manifest = {
        "pack": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "max_out_chars": cfg["max_out"],
        "config": cfg,
        "by_source": dict(by_source),
        "by_subset": dict(by_subset.most_common()),
        "ingest_stats": dict(stats),
        "files": {
            "alpaca_dingo": str(alpaca_path),
            "messages": str(messages_path),
        },
        "recommended_train": cfg["recommended_train"],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


PACKS = {
    "theory_gemma12b": {
        "max_out": 6000,
        "reasoning": "r1",  # one reasoning teacher to limit dupes
        "include_cli": True,
        "skills_cap": 2500,
        "ctftime_n": 2500,
        "kev_n": 800,
        "recommended_train": {
            "base_model": "mlx-community/gemma-4-12b-it-bf16",
            "max_seq_length": 4096,
            "iters": 1000,
            "notes": "Theory expert LoRA for Gemma 4 12B. Prefer shorter seq; multimodal base intact.",
        },
    },
    "theory_gemma26b": {
        "max_out": 12000,
        "reasoning": "both",
        "include_cli": True,
        "skills_cap": 4032,  # all skills
        "ctftime_n": 5000,
        "kev_n": 1500,
        "recommended_train": {
            "base_model": "mlx-community/gemma-4-26b-a4b-it-bf16",
            "max_seq_length": 6144,
            "iters": 1200,
            "notes": "Theory expert LoRA for Gemma 4 26B-A4B MoE. Longer writeups + both reasoning sets.",
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gemma 12B/26B theory SFT packs")
    parser.add_argument(
        "--pack",
        choices=["all", *PACKS.keys()],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    targets = list(PACKS.keys()) if args.pack == "all" else [args.pack]
    for name in targets:
        print(f"Building {name}...")
        man = build_pack(name, PACKS[name], seed=args.seed)
        print(f"  rows={man['rows']} -> {man['files']['alpaca_dingo']}")
        print(f"  sources={man['by_source']}")


if __name__ == "__main__":
    main()
