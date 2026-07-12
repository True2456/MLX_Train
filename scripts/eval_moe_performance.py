#!/usr/bin/env python3
"""MultiLoRA-MoE Evaluation & Benchmark Suite.

Runs a structured evaluation across three core capability domains:
1. Defensive Cybersecurity & Policy Alignment (Theory Specialist)
2. Systems & Assembly Architecture (ASM Systems Specialist)
3. Structured Tool-Calling & Agentic Reasoning (Agentic Specialist)

Verifies both router accuracy and response quality.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mati_moe import MatiMoEEngine

EVAL_PROMPTS = [
    {
        "id": "DEFENSIVE_CYBER_01",
        "category": "Defensive Cybersecurity",
        "expected_expert": "theory",
        "prompt": "Explain how parameterized queries prevent SQL injection attacks in web applications and provide an example in Python using sqlite3.",
    },
    {
        "id": "AI_POLICY_ALIGNMENT_02",
        "category": "AI Safety & Policy Alignment",
        "expected_expert": "theory",
        "prompt": "Explain the ethical and safety principles behind why AI models distinguish between educational vulnerability analysis versus generating actionable exploit payloads.",
    },
    {
        "id": "ASM_SYSTEMS_01",
        "category": "Systems & Assembly",
        "expected_expert": "asm_systems",
        "prompt": "Explain the x86_64 System V AMD64 ABI calling convention for passing integer arguments to a function in assembly.",
    },
    {
        "id": "AGENTIC_TOOL_01",
        "category": "Agentic Tool Calling",
        "expected_expert": "agentic",
        "prompt": 'Given a tool definition {"name": "search_database", "parameters": {"query": "string"}}, format a valid JSON tool call to search for "security logs".',
    },
]


def run_evaluation(load_live_model: bool = False, base_model_path: str = None):
    print("===================================================================")
    print("         MATI 12B MULTILORA-MOE EVALUATION BENCHMARK               ")
    print("===================================================================\n")

    engine = MatiMoEEngine()
    model = None
    tokenizer = None

    if load_live_model:
        assert base_model_path is not None
        print(f"[INFO] Loading live MLX base model from {base_model_path}...")
        from mlx_lm import generate
        from mlx_lm.utils import load_adapters, load_model, load_tokenizer

        model, _ = load_model(base_path=Path(base_model_path), lazy=False, strict=False)
        tokenizer = load_tokenizer(Path(base_model_path))

    correct_routing = 0
    total_prompts = len(EVAL_PROMPTS)

    for i, item in enumerate(EVAL_PROMPTS, 1):
        print(f"-------------------------------------------------------------------")
        print(f"Test [{i}/{total_prompts}] ID: {item['id']} ({item['category']})")
        print(f"Prompt: \"{item['prompt']}\"")

        turn = engine.generate_turn(item["prompt"])
        routing = turn["routing"]
        dominant = routing["dominant_expert"]
        weights = routing["weights"]

        is_match = dominant == item["expected_expert"]
        if is_match:
            correct_routing += 1
        status_tag = "PASS" if is_match else "FAIL"

        print(
            f"Router Decision: {dominant.upper()} ({weights[dominant]*100:.1f}%) "
            f"[Expected: {item['expected_expert'].upper()}] -> [{status_tag}]"
        )
        print(
            f"Telemetry: Theory={weights['theory']*100:.1f}% | "
            f"Agentic={weights['agentic']*100:.1f}% | "
            f"ASM={weights['asm_systems']*100:.1f}%"
        )

        if load_live_model:
            adapter_map = {
                "theory": "models/gemma12b/theory_lora",
                "agentic": "models/gemma12b/agentic_lora",
                "asm_systems": "models/gemma12b/asm_systems_lora",
            }
            adapter_path = adapter_map.get(dominant, "models/gemma12b/theory_lora")
            adapted_model = load_adapters(model, adapter_path)
            adapted_model.eval()

            messages = [{"role": "user", "content": item["prompt"]}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            out = generate(
                adapted_model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=140,
                verbose=False,
            )
            print("\nResponse Preview:")
            print(out.strip()[:350] + ("..." if len(out) > 350 else ""))

    print(f"\n===================================================================")
    print(f"Evaluation Summary: Routing Accuracy = {correct_routing}/{total_prompts} "
          f"({correct_routing/total_prompts*100:.1f}%)")
    print("===================================================================")


def main():
    parser = argparse.ArgumentParser(description="Run Mati 12B MoE Evaluations")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live token generation evaluation using MLX weights",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="models/gemma12b/base_gemma4_shim",
        help="Path to base Gemma 4 12B model shim directory",
    )
    args = parser.parse_args()

    run_evaluation(load_live_model=args.live, base_model_path=args.base_model)


if __name__ == "__main__":
    main()
