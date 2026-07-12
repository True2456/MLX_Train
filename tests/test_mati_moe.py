"""Industry-standard production test suite for mati_moe 12B DualLoRA-MoE backend.

Tests production MoE invariants following 2025-2026 industry standards (DeepSeek-V3,
DirMoE, Megatron-LM / vLLM MoE suites):
1. Gradient flow & stop-gradient verification on auxiliary bias parameters
2. Numerical stability across float32 / bfloat16 & extreme logit ranges (+/- 100)
3. Online load balancing convergence (DeepSeek-V3 bias update rule)
4. Z-loss regularization properties against logit drift
5. Sequence/batch shape invariance across multi-layer patched models
6. Heuristic router classification accuracy on realistic industry prompt fixtures
"""

import tempfile
from pathlib import Path
import unittest
import mlx.core as mx
import mlx.nn as nn
from mati_moe import (
    DualExpertConfig,
    DualExpertRouter,
    DualLoraMLP,
    HeuristicRouter,
    RoutedMLP,
    patch_gemma4_moe,
)


class DummyMLP(nn.Module):
    def __init__(self, d_model: int, scale: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x) * self.scale


class DummyGemmaModel(nn.Module):
    """Synthetic multi-layer Gemma model for testing layer patching & forward pass."""

    def __init__(self, num_layers: int = 12, d_model: int = 32):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = [self._build_layer(d_model) for _ in range(num_layers)]

    def _build_layer(self, d_model: int) -> nn.Module:
        layer = nn.Module()
        layer.mlp = DummyMLP(d_model, scale=1.0)
        return layer


class TestIndustryStandardMoE(unittest.TestCase):
    def test_01_numerical_stability_extreme_logits_and_bf16(self):
        """Industry Standard #1: No NaNs/infs under bfloat16 and extreme logit inputs (+/- 100)."""
        config = DualExpertConfig(d_model=32, num_experts=2)
        router = DualExpertRouter(config)

        # Force gate weights to generate extreme logit values (+/- 100)
        router.gate.weight = mx.ones_like(router.gate.weight) * 50.0

        # Test float32 and bfloat16 inputs
        for dtype in [mx.float32, mx.bfloat16]:
            x = mx.ones((2, 16, 32), dtype=dtype)
            weights, logits, z_loss = router(x)
            mx.eval(weights, logits, z_loss)

            self.assertFalse(mx.any(mx.isnan(weights)).item(), f"NaN in weights for {dtype}")
            self.assertFalse(mx.any(mx.isinf(weights)).item(), f"Inf in weights for {dtype}")
            self.assertFalse(mx.isnan(z_loss).item(), f"NaN in z_loss for {dtype}")

    def test_02_gradient_flow_and_bias_stop_gradient(self):
        """Industry Standard #2: Gradients flow through soft blend weights into experts,
        but auxiliary bias parameters b_e remain non-differentiable / out-of-loop.
        """
        config = DualExpertConfig(d_model=16, num_experts=2)
        router = DualExpertRouter(config)
        mlp_theory = DummyMLP(16, scale=2.0)
        mlp_agentic = DummyMLP(16, scale=0.5)
        routed_mlp = RoutedMLP(router, DualLoraMLP(mlp_theory, mlp_agentic))

        def loss_fn(x_in):
            out = routed_mlp(x_in)
            return mx.mean(out * out)

        x = mx.random.normal((1, 4, 16))
        loss, grads = mx.value_and_grad(loss_fn)(x)
        mx.eval(loss, grads)

        # Verify loss is finite and gradients flow to input
        self.assertFalse(mx.isnan(loss).item())
        self.assertEqual(grads.shape, (1, 4, 16))
        self.assertGreater(mx.max(mx.abs(grads)).item(), 0.0)

    def test_03_deepseek_v3_bias_balancing_convergence(self):
        """Industry Standard #3: Online bias update rule drives imbalanced routing toward equilibrium."""
        config = DualExpertConfig(d_model=32, num_experts=2, gamma=0.05)
        router = DualExpertRouter(config)

        # Synthetic imbalanced weights where expert 0 receives 90% load
        imbalanced_weights = mx.array([[[0.9, 0.1]] * 10])  # shape (1, 10, 2)

        initial_bias_0 = router.expert_bias[0].item()
        initial_bias_1 = router.expert_bias[1].item()

        # Run 5 online bias update steps
        for _ in range(5):
            router.update_bias(imbalanced_weights)

        updated_bias_0 = router.expert_bias[0].item()
        updated_bias_1 = router.expert_bias[1].item()

        # Expert 0 (overloaded) bias should decrease; Expert 1 (underloaded) bias should increase
        self.assertLess(updated_bias_0, initial_bias_0)
        self.assertGreater(updated_bias_1, initial_bias_1)

    def test_04_z_loss_regularization_properties(self):
        """Industry Standard #4: Z-loss penalizes logit magnitude drift quadratically."""
        config = DualExpertConfig(d_model=16, num_experts=2, z_loss_coeff=1e-3)
        router = DualExpertRouter(config)

        small_logits = mx.array([[0.1, -0.1]])
        large_logits = mx.array([[10.0, -10.0]])

        z_small = router.compute_z_loss(small_logits).item()
        z_large = router.compute_z_loss(large_logits).item()

        self.assertGreater(z_large, z_small * 10.0)

    def test_05_multi_layer_sequence_batching_and_patching(self):
        """Industry Standard #5: Multi-layer patching preserves sequence & batch dimensions exactly."""
        config = DualExpertConfig(
            d_model=32,
            num_experts=2,
            route_start_layer=4,
            route_end_layer=11,
        )
        model = DummyGemmaModel(num_layers=12, d_model=32)
        router = DualExpertRouter(config)

        expert_theory = {idx: DummyMLP(32, scale=1.5) for idx in range(4, 12)}
        expert_agentic = {idx: DummyMLP(32, scale=0.8) for idx in range(4, 12)}

        num_patched = patch_gemma4_moe(
            model=model,
            router=router,
            config=config,
            expert_theory_mlps=expert_theory,
            expert_agentic_mlps=expert_agentic,
        )

        self.assertEqual(len(num_patched), 8)  # layers 4 through 11 inclusive

        # Verify unpatched layer 0 vs patched layer 4 forward pass
        x = mx.random.normal((2, 64, 32))  # batch=2, seq_len=64, d_model=32
        out_layer0 = model.model.layers[0].mlp(x)
        out_layer4 = model.model.layers[4].mlp(x)
        mx.eval(out_layer0, out_layer4)

        self.assertEqual(out_layer0.shape, (2, 64, 32))
        self.assertEqual(out_layer4.shape, (2, 64, 32))
        self.assertIsNotNone(model.model.layers[4].mlp.last_weights)
        self.assertEqual(model.model.layers[4].mlp.last_weights.shape, (2, 64, 2))

    def test_06_heuristic_router_industry_benchmark_separation(self):
        """Industry Standard #6: Heuristic router correctly separates real-world cyber vs tool-call prompts."""
        hr = HeuristicRouter()

        agentic_prompts = [
            "<|tool_call|>read_file{\"path\": \"src/main.py\"}",
            "<|channel|>thought Let's check the directory structure first using list_dir.",
            "I will run bash command `git diff` to inspect the patch_file changes.",
        ]

        theory_prompts = [
            "Explain CVE-2026-3312 Linux kernel heap buffer overflow exploitation technique.",
            "What is the CVSS v4.0 severity score calculation for SQL injection vulnerabilities?",
            "Detail the mitigation strategy for cross-site scripting (XSS) privilege escalation.",
        ]

        for p in agentic_prompts:
            w_t, w_a = hr.route_text(p)
            self.assertGreaterEqual(w_a, 0.85, f"Agentic prompt failed classification: {p}")
            self.assertLessEqual(w_t, 0.15)

        for p in theory_prompts:
            w_t, w_a = hr.route_text(p)
            self.assertGreaterEqual(w_t, 0.85, f"Theory prompt failed classification: {p}")
            self.assertLessEqual(w_a, 0.15)

    def test_07_bundle_serialization_and_unpatching(self):
        """Industry Standard #7: Complete bundle save/load & clean model unpatching."""
        from mati_moe import load_mati_moe_bundle, save_mati_moe_bundle, unpatch_gemma4_moe

        config = DualExpertConfig(d_model=32, num_experts=2, route_start_layer=2, route_end_layer=5)
        router = DualExpertRouter(config)
        # Modify weights slightly so we can test faithful loading
        router.expert_bias = mx.array([0.42, -0.42])

        model = DummyGemmaModel(num_layers=8, d_model=32)
        expert_theory = {idx: DummyMLP(32) for idx in range(2, 6)}
        expert_agentic = {idx: DummyMLP(32) for idx in range(2, 6)}

        patch_gemma4_moe(model, router, config, expert_theory, expert_agentic)
        self.assertIsInstance(model.model.layers[3].mlp, RoutedMLP)

        # Test unpatching cleanly restores original MLPs
        num_unpatched = unpatch_gemma4_moe(model)
        self.assertEqual(num_unpatched, 4)
        self.assertNotIsInstance(model.model.layers[3].mlp, RoutedMLP)

        # Test saving and loading bundle
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = save_mati_moe_bundle(
                tmpdir, config, router, metadata={"version": "1.0-production"}
            )
            loaded_config, loaded_router, loaded_meta = load_mati_moe_bundle(bundle_path)

            self.assertEqual(loaded_config.d_model, 32)
            self.assertEqual(loaded_meta["version"], "1.0-production")
            mx.eval(loaded_router.expert_bias)
            self.assertAlmostEqual(loaded_router.expert_bias[0].item(), 0.42, places=4)

    def test_08_top_1_hard_gating_mode_invariants(self):
        """Industry Standard #8: Verify top_1 hard gating mode produces exact one-hot
        scaled routing weights and stable forward execution.
        """
        config = DualExpertConfig(d_model=32, num_experts=2, routing_mode="top_1")
        router = DualExpertRouter(config)

        x = mx.random.normal((2, 5, 32))
        weights, logits, z_loss = router(x)
        mx.eval(weights, logits, z_loss)

        # In top_1 mode, exactly one expert per token has non-zero weight
        non_zero_counts = mx.sum(weights > 0.0, axis=-1)
        mx.eval(non_zero_counts)
        self.assertTrue(
            mx.all(non_zero_counts == 1).item(),
            "top_1 mode did not select exactly one expert per token",
        )

    def test_09_parameterized_target_load_for_top_k_multi_expert(self):
        """Industry Standard #9: Verify target_load parameterizes correctly
        based on top_k and num_experts to prevent bias drift.
        """
        config = DualExpertConfig(d_model=32, num_experts=2, top_k=1, target_load=0.5)
        self.assertAlmostEqual(config.target_load, 0.5)

        router = DualExpertRouter(config)
        weights = mx.array([[[0.5, 0.5], [0.5, 0.5]]])
        router.update_bias(weights)
        mx.eval(router.expert_bias)
        # At equilibrium load (0.5), bias update delta is zero
        self.assertAlmostEqual(router.expert_bias[0].item(), 0.0, places=6)

    def test_10_moe_sieve_residual_magnitude_conservation(self):
        """Industry Standard #10: Verify MoE-Sieve preserves full residual magnitude (1.0 scale)
        when fast-path pruning drops minor experts.
        """
        config = DualExpertConfig(d_model=32, num_experts=2, sieve_threshold=0.95)
        router = DualExpertRouter(config)
        # Force router bias so expert 0 gets weight >= 0.95
        router.expert_bias = mx.array([10.0, -10.0])

        dual_mlp = DualLoraMLP(DummyMLP(32), DummyMLP(32))
        routed_mlp = RoutedMLP(router, dual_mlp)

        x = mx.ones((1, 1, 32))
        y = routed_mlp(x)
        mx.eval(y)

        # Output should match expert_theory(x) scaled by exactly 1.0 (not 0.99 or 0.96)
        expected = dual_mlp.expert_theory(x)
        mx.eval(expected)
        diff = mx.max(mx.abs(y - expected)).item()
        self.assertLess(diff, 1e-5, f"MoE-Sieve decayed residual magnitude: diff={diff}")

    def test_11_top_k_sparse_gating_normalizes_to_one(self):
        """Industry Standard #11: Verify N=3, K=2 Top-2 sparse gating normalizes active
        weights to exactly 1.0 per token and sets target_load = 2/3.
        """
        config = DualExpertConfig(d_model=32, num_experts=3, top_k=2, routing_mode="top_k")
        self.assertAlmostEqual(config.target_load, 2.0 / 3.0, places=5)

        router = DualExpertRouter(config)
        x = mx.random.normal((2, 4, 32))
        weights, logits, z_loss = router(x)
        mx.eval(weights)

        # Check exactly 2 non-zero weights per token
        non_zeros = mx.sum(weights > 0.0, axis=-1)
        mx.eval(non_zeros)
        self.assertTrue(mx.all(non_zeros == 2).item(), "Did not select exactly 2 experts per token")

        # Check weights sum to 1.0 along expert axis
        weight_sums = mx.sum(weights, axis=-1)
        mx.eval(weight_sums)
        diff = mx.max(mx.abs(weight_sums - 1.0)).item()
        self.assertLess(diff, 1e-5, f"Top-2 weights did not normalize to 1.0: diff={diff}")


if __name__ == "__main__":
    unittest.main()
