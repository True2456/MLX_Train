"""Multimodal (Vision & Audio) Subspace Stability Test Suite for 12B DualLoRA-MoE.

In Gemma 4 12B Unified (encoder-free multimodal architecture), vision, audio, and text tokens
share unified internal representations across layers 0..47.

This test suite verifies:
1. Multimodal Token Routing Stability: Vision & audio token representations passing through
   RoutedMLP do not produce NaNs, Infs, or shape corruption.
2. Visual-Linguistic Subspace Preservation: Quantifies cosine similarity between un-adapted
   base MLP outputs and DualLoRA-MoE blended outputs on synthetic vision/audio feature vectors
   to guard against "intruder dimension" drift.
3. Multimodal Prompt Routing Separation: Verifies that multimodal screenshot/OCR agentic prompts
   vs multimodal technical diagram prompts route to the appropriate specialist.
"""

import unittest
import mlx.core as mx
import mlx.nn as nn
from mati_moe import (
    DualExpertConfig,
    DualExpertRouter,
    DualLoraMLP,
    HeuristicRouter,
    RoutedMLP,
)


class DummyBaseMLP(nn.Module):
    """Simulates Gemma 4 base MLP block."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class DummyLoRAMLP(nn.Module):
    """Simulates Base MLP + LoRA delta specialist."""
    def __init__(self, base_mlp: DummyBaseMLP, lora_scale: float = 0.1):
        super().__init__()
        self.base_mlp = base_mlp
        self.lora_A = nn.Linear(base_mlp.proj.weight.shape[1], 4, bias=False)
        self.lora_B = nn.Linear(4, base_mlp.proj.weight.shape[0], bias=False)
        self.scale = lora_scale

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base_mlp(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scale
        return base_out + lora_out


class TestMultimodalMoE(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.base_mlp = DummyBaseMLP(self.d_model)
        self.expert_theory = DummyLoRAMLP(self.base_mlp, lora_scale=0.05)
        self.expert_agentic = DummyLoRAMLP(self.base_mlp, lora_scale=0.05)

        self.config = DualExpertConfig(d_model=self.d_model, num_experts=2)
        self.router = DualExpertRouter(self.config)
        self.routed_mlp = RoutedMLP(
            router=self.router,
            dual_mlp=DualLoraMLP(self.expert_theory, self.expert_agentic),
        )

    def test_01_vision_and_audio_token_embedding_forward_pass(self):
        """Verify continuous vision & audio feature tokens pass through RoutedMLP without errors."""
        # Synthetic multimodal sequence: [B=2, S=32 (10 text + 16 vision + 6 audio), D=64]
        multimodal_seq = mx.random.normal((2, 32, self.d_model))

        out = self.routed_mlp(multimodal_seq)
        mx.eval(out)

        self.assertEqual(out.shape, (2, 32, self.d_model))
        self.assertFalse(mx.any(mx.isnan(out)).item())
        self.assertFalse(mx.any(mx.isinf(out)).item())

    def test_02_visual_linguistic_subspace_drift_guard(self):
        """Verify DualLoRA-MoE blended output maintains >= 0.95 cosine similarity with base MLP
        on visual and audio token representations (no severe subspace distortion).
        """
        # Synthetic visual/audio patches (high-norm continuous representations)
        vision_tokens = mx.random.normal((1, 16, self.d_model)) * 2.0

        base_out = self.base_mlp(vision_tokens)
        moe_out = self.routed_mlp(vision_tokens)

        # Compute cosine similarity across hidden dimension
        dot = mx.sum(base_out * moe_out, axis=-1)
        norm_base = mx.sqrt(mx.sum(base_out * base_out, axis=-1))
        norm_moe = mx.sqrt(mx.sum(moe_out * moe_out, axis=-1))
        cos_sim = mx.mean(dot / (norm_base * norm_moe + 1e-8)).item()

        # LoRA rank adaptations with small scale should preserve >= 95% alignment with base visual path
        self.assertGreaterEqual(
            cos_sim,
            0.95,
            f"Severe visual subspace drift detected: cosine sim = {cos_sim:.4f} < 0.95",
        )

    def test_03_multimodal_agentic_screenshot_vs_diagram_routing(self):
        """Verify HeuristicRouter routes multimodal screenshot/OCR tasks vs technical diagrams correctly."""
        hr = HeuristicRouter()

        agentic_vision_prompt = (
            "<|image|> Look at this screenshot and run <|tool_call|>bash "
            "to execute the command shown on screen."
        )
        theory_vision_prompt = (
            "<|image|> Analyze this network architecture diagram and explain the "
            "CVE-2026-4401 privilege escalation path."
        )

        w_t1, w_a1 = hr.route_text(agentic_vision_prompt)
        self.assertGreaterEqual(w_a1, 0.85, "Screenshot tool-call prompt didn't route to agentic")

        w_t2, w_a2 = hr.route_text(theory_vision_prompt)
        self.assertGreaterEqual(w_t2, 0.85, "CVE diagram prompt didn't route to theory")


if __name__ == "__main__":
    unittest.main()
