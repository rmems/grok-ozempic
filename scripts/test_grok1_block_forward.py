#!/usr/bin/env python3
"""Tests for the Grok-1 block forward kernels (GH #61 / RM-249).

The kernels are checked against independently written naive references and
against hand-derived invariants, not just shapes: an einsum index slip or a
GELU on the wrong GeGLU branch is shape-legal and would silently corrupt every
route-preservation number downstream.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grok1_block_forward import (  # noqa: E402
    ATTN_OUTPUT_MULTIPLIER,
    EMBEDDING_MULTIPLIER,
    FFN_SIZE,
    GROK1_HEADS,
    KEY_SIZE,
    MAX_ATTN_VAL,
    MODEL_SIZE,
    NUM_EXPERTS,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    SLOT_ROLES,
    ForwardError,
    HeadConfig,
    attention,
    causal_mask,
    expert_geglu,
    gelu,
    moe_combine,
    resolve_roles,
    rmsnorm,
    rope,
    rotate_half,
    router_logits,
    top_k_experts,
)


def _block_shapes() -> dict[str, tuple[int, ...]]:
    """Real block-0 structural names and shapes, as xai-dissect reports them."""
    return {
        "block_000.slot_00.moe_expert.gate": (NUM_EXPERTS, MODEL_SIZE, FFN_SIZE),
        "block_000.slot_01.moe_expert.down": (NUM_EXPERTS, FFN_SIZE, MODEL_SIZE),
        "block_000.slot_02.moe_expert.up": (NUM_EXPERTS, MODEL_SIZE, FFN_SIZE),
        "block_000.slot_03.attn_proj_i8.narrow": (MODEL_SIZE, NUM_KV_HEADS * KEY_SIZE),
        "block_000.slot_04.attn_proj_i8.model_width": (MODEL_SIZE, MODEL_SIZE),
        "block_000.slot_05.attn_proj_i8.model_width": (MODEL_SIZE, NUM_Q_HEADS * KEY_SIZE),
        "block_000.slot_06.attn_proj_i8.narrow": (MODEL_SIZE, NUM_KV_HEADS * KEY_SIZE),
        "block_000.slot_07.block_norm": (MODEL_SIZE,),
        "block_000.slot_08.block_norm": (MODEL_SIZE,),
        "block_000.slot_09.block_norm": (MODEL_SIZE,),
        "block_000.slot_10.block_norm": (MODEL_SIZE,),
        "block_000.slot_11.router": (MODEL_SIZE, NUM_EXPERTS),
    }


class SlotRoleTests(unittest.TestCase):
    def test_real_block_shapes_resolve_to_every_role(self):
        roles = resolve_roles(_block_shapes())
        self.assertEqual(roles["query"], "block_000.slot_05.attn_proj_i8.model_width")
        self.assertEqual(roles["attn_out"], "block_000.slot_04.attn_proj_i8.model_width")
        self.assertEqual(roles["key"], "block_000.slot_03.attn_proj_i8.narrow")
        self.assertEqual(roles["value"], "block_000.slot_06.attn_proj_i8.narrow")

    def test_router_input_is_slot_09_and_post_attention_norm_is_slot_08(self):
        """The #57 ambiguity: (post=08, pre=09), per DecoderLayer call order."""
        self.assertEqual(SLOT_ROLES["slot_08"], "norm_post_attn")
        self.assertEqual(SLOT_ROLES["slot_09"], "norm_pre_moe")

    def test_gelu_branch_is_slot_00_and_passthrough_is_slot_02(self):
        """MoELayer applies gelu to params["linear"] == slot_00, not slot_02."""
        self.assertEqual(SLOT_ROLES["slot_00"], "expert_gelu")
        self.assertEqual(SLOT_ROLES["slot_02"], "expert_value")

    def test_swapped_query_and_output_projection_is_rejected_by_shape(self):
        shapes = _block_shapes()
        shapes["block_000.slot_05.attn_proj_i8.model_width"] = (MODEL_SIZE, 999)
        with self.assertRaises(ForwardError):
            resolve_roles(shapes)

    def test_missing_tensor_is_rejected(self):
        shapes = _block_shapes()
        del shapes["block_000.slot_11.router"]
        with self.assertRaisesRegex(ForwardError, "missing required tensors"):
            resolve_roles(shapes)

    def test_unknown_slot_is_rejected(self):
        shapes = {**_block_shapes(), "block_000.slot_12.mystery": (MODEL_SIZE,)}
        with self.assertRaisesRegex(ForwardError, "unknown slot"):
            resolve_roles(shapes)

    def test_name_without_slot_component_is_rejected(self):
        with self.assertRaisesRegex(ForwardError, "no slot_NN component"):
            resolve_roles({"embedding.token_embedding": (MODEL_SIZE,)})

    def test_duplicate_role_is_rejected(self):
        shapes = _block_shapes()
        shapes["block_001.slot_11.router"] = (MODEL_SIZE, NUM_EXPERTS)
        with self.assertRaisesRegex(ForwardError, "claimed by both"):
            resolve_roles(shapes)


class RmsNormTests(unittest.TestCase):
    def test_matches_hand_computed_value(self):
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        gain = np.array([2.0, 0.5], dtype=np.float32)
        ms = (9.0 + 16.0) / 2.0
        expect = np.array([[3.0, 4.0]]) / math.sqrt(ms + 1e-5) * np.array([2.0, 0.5])
        np.testing.assert_allclose(rmsnorm(x, gain), expect, rtol=1e-6)

    def test_is_invariant_to_a_global_input_scale(self):
        """rmsnorm(c*x) == rmsnorm(x); why the embedding multiplier is irrelevant."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 64)).astype(np.float32)
        gain = rng.standard_normal(64).astype(np.float32)
        np.testing.assert_allclose(rmsnorm(78.38 * x, gain), rmsnorm(x, gain), rtol=2e-4)

    def test_applies_gain_as_plain_multiply_not_one_plus_gain(self):
        """Upstream is ``scale * normed``, despite scale initializing to zero."""
        x = np.ones((1, 4), dtype=np.float32)
        self.assertAlmostEqual(float(rmsnorm(x, np.zeros(4, dtype=np.float32)).sum()), 0.0)


class RopeTests(unittest.TestCase):
    def test_rotate_half_negates_the_second_half(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        np.testing.assert_allclose(rotate_half(x), [[-3.0, -4.0, 1.0, 2.0]])

    def test_position_zero_is_the_identity(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 2, 8)).astype(np.float32)
        np.testing.assert_allclose(rope(x), x, rtol=1e-6, atol=1e-6)

    def test_preserves_per_head_norm(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((6, 3, 16)).astype(np.float32)
        before = np.linalg.norm(x, axis=-1)
        after = np.linalg.norm(rope(x), axis=-1)
        np.testing.assert_allclose(after, before, rtol=1e-5)

    def test_matches_naive_paired_rotation(self):
        """Half-split convention: dim d pairs with d + D/2 under one phase."""
        dim, tokens = 8, 5
        rng = np.random.default_rng(3)
        x = rng.standard_normal((tokens, 1, dim)).astype(np.float32)
        got = rope(x)
        half = dim // 2
        for t in range(tokens):
            for j in range(half):
                freq = 1.0 / (10000.0 ** ((2 * j) / dim))
                cos, sin = math.cos(t * freq), math.sin(t * freq)
                a, b = float(x[t, 0, j]), float(x[t, 0, j + half])
                self.assertAlmostEqual(float(got[t, 0, j]), a * cos - b * sin, places=4)
                self.assertAlmostEqual(float(got[t, 0, j + half]), b * cos + a * sin, places=4)

    def test_offset_shifts_positions(self):
        """``offset=n`` on row 0 must equal position ``n`` applied to the same row.

        The rows are held identical on purpose: comparing ``rope(x, offset=2)[0]``
        against ``rope(x)[2]`` would compare phases applied to *different* inputs.
        """
        rng = np.random.default_rng(4)
        row = rng.standard_normal((1, 1, 8)).astype(np.float32)
        same = np.repeat(row, 3, axis=0)
        np.testing.assert_allclose(
            rope(row, offset=2)[0], rope(same)[2], rtol=1e-5, atol=1e-6
        )

    def test_odd_head_dim_is_rejected(self):
        with self.assertRaisesRegex(ForwardError, "even head dim"):
            rope(np.zeros((1, 1, 5), dtype=np.float32))


def _naive_attention(hn, wq, wk, wv, wo, n_q, n_kv, hd):
    """Independent per-head loop reference for the grouped-query einsums."""
    tokens = hn.shape[0]
    groups = n_q // n_kv
    q = rope((hn @ wq).reshape(tokens, n_q, hd))
    k = rope((hn @ wk).reshape(tokens, n_kv, hd))
    v = (hn @ wv).reshape(tokens, n_kv, hd)
    ctx = np.zeros((tokens, n_q, hd), dtype=np.float32)
    for qh in range(n_q):
        kvh = qh // groups  # contiguous grouping
        logits = (q[:, qh, :] @ k[:, kvh, :].T) * ATTN_OUTPUT_MULTIPLIER
        logits = MAX_ATTN_VAL * np.tanh(logits / MAX_ATTN_VAL)
        logits = np.where(causal_mask(tokens), logits, -1e30)
        logits -= np.max(logits, axis=-1, keepdims=True)
        w = np.exp(logits)
        w /= np.sum(w, axis=-1, keepdims=True)
        ctx[:, qh, :] = w @ v[:, kvh, :]
    return ctx.reshape(tokens, n_q * hd) @ wo


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.n_q, self.n_kv, self.hd, self.tokens = 6, 3, 4, 5
        self.heads = HeadConfig(n_q_heads=self.n_q, n_kv_heads=self.n_kv, head_dim=self.hd)
        self.width = 8
        rng = np.random.default_rng(10)
        self.hn = rng.standard_normal((self.tokens, self.width)).astype(np.float32)
        self.wq = rng.standard_normal((self.width, self.n_q * self.hd)).astype(np.float32)
        self.wk = rng.standard_normal((self.width, self.n_kv * self.hd)).astype(np.float32)
        self.wv = rng.standard_normal((self.width, self.n_kv * self.hd)).astype(np.float32)
        self.wo = rng.standard_normal((self.n_q * self.hd, self.width)).astype(np.float32)

    def _call(self, **kw):
        return attention(
            self.hn, self.wq, self.wk, self.wv, self.wo,
            heads=self.heads, **kw,
        )

    def test_matches_naive_per_head_reference(self):
        expect = _naive_attention(
            self.hn, self.wq, self.wk, self.wv, self.wo, self.n_q, self.n_kv, self.hd
        )
        np.testing.assert_allclose(self._call(), expect, rtol=1e-4, atol=1e-4)

    def test_is_causal_first_token_ignores_the_future(self):
        """Perturbing a later token must not change the first token's output."""
        base = self._call()
        bumped = self.hn.copy()
        bumped[-1] += 5.0
        moved = attention(
            bumped, self.wq, self.wk, self.wv, self.wo,
            heads=self.heads,
        )
        np.testing.assert_allclose(moved[0], base[0], rtol=1e-5, atol=1e-5)
        self.assertGreater(float(np.abs(moved[-1] - base[-1]).max()), 1e-6)

    def test_grouped_heads_share_key_value_heads(self):
        """Zeroing kv head 0 changes only the q heads in its group."""
        wk2, wv2 = self.wk.copy(), self.wv.copy()
        wk2[:, : self.hd] = 0.0
        wv2[:, : self.hd] = 0.0
        changed = attention(
            self.hn, self.wq, wk2, wv2, np.eye(self.n_q * self.hd, dtype=np.float32),
            heads=self.heads,
        ).reshape(self.tokens, self.n_q, self.hd)
        base = attention(
            self.hn, self.wq, self.wk, self.wv, np.eye(self.n_q * self.hd, dtype=np.float32),
            heads=self.heads,
        ).reshape(self.tokens, self.n_q, self.hd)
        groups = self.n_q // self.n_kv
        self.assertGreater(float(np.abs(changed[:, :groups] - base[:, :groups]).max()), 1e-6)
        np.testing.assert_allclose(changed[:, groups:], base[:, groups:], rtol=1e-5, atol=1e-5)

    def test_soft_cap_bounds_extreme_logits(self):
        """A huge projection must not produce a degenerate one-hot softmax."""
        big = self.wq * 1e6
        out = attention(
            self.hn, big, self.wk, self.wv, self.wo,
            heads=self.heads,
        )
        self.assertTrue(np.isfinite(out).all())

    def test_head_count_must_divide(self):
        with self.assertRaisesRegex(ForwardError, "not a multiple"):
            HeadConfig(n_q_heads=5, n_kv_heads=2, head_dim=self.hd)

    def test_real_head_config_matches_grok1(self):
        self.assertEqual(GROK1_HEADS.n_q_heads, NUM_Q_HEADS)
        self.assertEqual(GROK1_HEADS.n_kv_heads, NUM_KV_HEADS)
        self.assertEqual(GROK1_HEADS.head_dim, KEY_SIZE)
        self.assertEqual(GROK1_HEADS.groups, 6)


class GeluTests(unittest.TestCase):
    def test_matches_the_jax_tanh_formulation(self):
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        np.testing.assert_allclose(gelu(x), 0.5 * x * (1.0 + np.tanh(inner)), rtol=1e-6)

    def test_differs_measurably_from_the_erf_form(self):
        """Confirms the tanh/erf choice is not cosmetic."""
        x = np.linspace(-3, 3, 401).astype(np.float32)
        exact = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        self.assertGreater(float(np.abs(gelu(x) - exact).max()), 1e-4)

    def test_zero_maps_to_zero(self):
        self.assertEqual(float(gelu(np.zeros(1, dtype=np.float32))[0]), 0.0)


class RoutingTests(unittest.TestCase):
    def test_router_logits_are_a_plain_matmul(self):
        rng = np.random.default_rng(20)
        h = rng.standard_normal((3, 5)).astype(np.float32)
        w = rng.standard_normal((5, 4)).astype(np.float32)
        np.testing.assert_allclose(router_logits(h, w), h @ w, rtol=1e-5)

    def test_gates_are_raw_softmax_and_not_renormalized(self):
        logits = np.array([[3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
        idx, gates = top_k_experts(logits, k=2)
        np.testing.assert_array_equal(idx, [[0, 1]])
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        np.testing.assert_allclose(gates, probs[:, :2], rtol=1e-5)
        self.assertLess(float(gates.sum()), 1.0)

    def test_indices_are_ordered_by_descending_probability(self):
        rng = np.random.default_rng(21)
        logits = rng.standard_normal((32, NUM_EXPERTS)).astype(np.float32)
        idx, gates = top_k_experts(logits, k=3)
        self.assertTrue(bool(np.all(np.diff(gates, axis=1) <= 1e-7)))
        top = logits.argmax(axis=1)
        np.testing.assert_array_equal(idx[:, 0], top)

    def test_top_k_below_one_is_rejected(self):
        """k=0 would yield an empty routing dimension, not an error."""
        logits = np.zeros((4, NUM_EXPERTS), dtype=np.float32)
        for bad in (0, -1):
            with self.assertRaisesRegex(ForwardError, "top_k must be in"):
                top_k_experts(logits, k=bad)

    def test_top_k_above_expert_count_is_rejected(self):
        """k>experts would silently route every token to every expert."""
        logits = np.zeros((4, NUM_EXPERTS), dtype=np.float32)
        with self.assertRaisesRegex(ForwardError, "top_k must be in"):
            top_k_experts(logits, k=NUM_EXPERTS + 1)

    def test_top_k_equal_to_expert_count_is_allowed(self):
        logits = np.zeros((4, NUM_EXPERTS), dtype=np.float32)
        idx, _ = top_k_experts(logits, k=NUM_EXPERTS)
        self.assertEqual(idx.shape, (4, NUM_EXPERTS))

    def test_selection_by_probability_equals_selection_by_logit(self):
        rng = np.random.default_rng(22)
        logits = rng.standard_normal((64, NUM_EXPERTS)).astype(np.float32)
        idx, _ = top_k_experts(logits, k=2)
        expect = np.argsort(-logits, axis=1, kind="stable")[:, :2]
        np.testing.assert_array_equal(idx, expect)


class ExpertTests(unittest.TestCase):
    def test_geglu_applies_gelu_to_the_gelu_branch_only(self):
        """Swapping the two branches changes the result -- the mapping matters."""
        rng = np.random.default_rng(30)
        h = rng.standard_normal((4, 6)).astype(np.float32)
        w_gelu = rng.standard_normal((6, 7)).astype(np.float32)
        w_val = rng.standard_normal((6, 7)).astype(np.float32)
        w_down = rng.standard_normal((7, 6)).astype(np.float32)
        correct = expert_geglu(h, w_gelu, w_val, w_down)
        swapped = expert_geglu(h, w_val, w_gelu, w_down)
        np.testing.assert_allclose(correct, (gelu(h @ w_gelu) * (h @ w_val)) @ w_down, rtol=1e-5)
        self.assertGreater(float(np.abs(correct - swapped).max()), 1e-5)

    def test_moe_combine_weights_by_gate_and_sums_over_selected_experts(self):
        idx = np.array([[0, 1], [1, 0]], dtype=np.int64)
        gates = np.array([[0.6, 0.3], [0.7, 0.2]], dtype=np.float32)
        # Expert 0 serves token 0 (slot 0) then token 1 (slot 1); expert 1 the reverse.
        outputs = {
            0: np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            1: np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32),
        }
        got = moe_combine(outputs, idx, gates, tokens=2)
        # token 0: 0.6*[1,0] (expert 0) + 0.3*[0,1] (expert 1, its slot 1)
        np.testing.assert_allclose(got[0], [0.6, 0.3], rtol=1e-6)
        # token 1: 0.2*[2,0] (expert 0 at slot 1) + 0.7*[0,2] (expert 1 at slot 0)
        np.testing.assert_allclose(got[1], [0.4, 1.4], rtol=1e-6)

    def test_moe_combine_rejects_a_row_count_mismatch(self):
        idx = np.array([[0, 1]], dtype=np.int64)
        gates = np.array([[0.5, 0.5]], dtype=np.float32)
        with self.assertRaisesRegex(ForwardError, "routed tokens but"):
            moe_combine({0: np.zeros((3, 2), dtype=np.float32)}, idx, gates, tokens=1)


class EmbeddingMultiplierTests(unittest.TestCase):
    """The residual add is not scale-invariant, unlike the pre-attention norm."""

    def test_multiplier_brings_grok1_embedding_rows_to_unit_rms(self):
        # Measured RMS of real ckpt-0 token-embedding rows.
        measured_rms = 0.0127436
        self.assertAlmostEqual(measured_rms * EMBEDDING_MULTIPLIER, 1.0, places=2)

    def test_residual_add_is_not_scale_invariant(self):
        """Why omitting the multiplier is nearly invisible in #57 yet fatal in #61.

        The residual add compares the sublayer delta against the stream magnitude
        directly, so a 78x too-small stream makes attention appear to dominate.
        """
        rng = np.random.default_rng(40)
        gain = np.ones(32, dtype=np.float32)
        h0 = rng.standard_normal((4, 32)).astype(np.float32) * np.float32(0.0127)
        delta = rmsnorm(rng.standard_normal((4, 32)).astype(np.float32), gain)
        unscaled = np.linalg.norm(delta) / np.linalg.norm(h0)
        scaled = np.linalg.norm(delta) / np.linalg.norm(h0 * np.float32(EMBEDDING_MULTIPLIER))
        self.assertGreater(unscaled / scaled, 70.0)

    def test_rmsnorm_scale_invariance_holds_only_when_eps_is_negligible(self):
        """``eps=1e-5`` breaks exact invariance at raw-embedding magnitudes.

        Raw Grok-1 embedding rows have RMS ~0.0127, so ``mean(x^2) ~= 1.6e-4`` and
        ``eps`` is ~6% of it -- a ~3% deviation. At unit RMS (post-multiplier) eps
        is negligible and invariance is tight. So even a norm-only proxy is
        mildly scale-dependent; it is the residual add that makes it decisive.
        """
        rng = np.random.default_rng(41)
        gain = np.ones(32, dtype=np.float32)
        unit = rng.standard_normal((4, 32)).astype(np.float32)
        np.testing.assert_allclose(rmsnorm(2.0 * unit, gain), rmsnorm(unit, gain), rtol=1e-5)

        tiny = unit * np.float32(0.0127)
        deviation = np.abs(
            rmsnorm(tiny * np.float32(EMBEDDING_MULTIPLIER), gain) / rmsnorm(tiny, gain) - 1.0
        ).max()
        self.assertGreater(float(deviation), 0.02)
        self.assertLess(float(deviation), 0.05)


class CausalMaskTests(unittest.TestCase):
    def test_is_lower_triangular(self):
        np.testing.assert_array_equal(
            causal_mask(3), [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
        )


if __name__ == "__main__":
    unittest.main()
