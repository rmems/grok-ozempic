#!/usr/bin/env python3
"""Genuine Grok-1 decoder-block forward pass (GH #61 / RM-249).

Every kernel here mirrors ``xai-org/grok-1`` ``model.py`` exactly; the module is
deliberately free of measurement or I/O policy so it can be unit-tested against
hand-computed values. Departures from upstream would silently invalidate every
route-preservation number built on top of it, so the mapping from checkpoint
slot to architectural role is asserted rather than assumed.

Upstream references (``xai-org/grok-1`` @ ``model.py`` / ``run.py``):

* ``DecoderLayer.__call__``   -- block body and the four RMSNorm call sites
* ``MultiHeadAttention``      -- projections, RoPE, tanh soft-cap, GQA einsums
* ``DenseBlock``              -- GeGLU expert feed-forward
* ``MoELayer._inference_call`` -- top-k selection and gate combination
* ``RMSNorm.__call__``        -- ``scale * x * rsqrt(mean(x^2) + eps)``

Block body, verbatim from ``DecoderLayer``::

    h = inputs
    h_attn = MHABlock(...)(layer_norm(h), ...)   # norm 1: pre-attention
    h_attn = layer_norm(h_attn)                  # norm 2: post-attention
    h += h_attn                                  # residual add
    h_dense = MoELayer(...)(layer_norm(h), ...)  # norm 3: pre-MoE == router input
    h_dense = layer_norm(h_dense)                # norm 4: post-MoE
    h += h_dense

Note that norms 2 and 4 are applied to the *sublayer output before* the residual
add, not to the residual stream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ATTN_OUTPUT_MULTIPLIER",
    "EMBEDDING_MULTIPLIER",
    "FFN_SIZE",
    "GROK1_HEADS",
    "HeadConfig",
    "KEY_SIZE",
    "MAX_ATTN_VAL",
    "MODEL_SIZE",
    "NUM_EXPERTS",
    "NUM_KV_HEADS",
    "NUM_Q_HEADS",
    "NUM_SELECTED_EXPERTS",
    "RMS_EPS",
    "ROPE_BASE",
    "SLOT_ROLES",
    "ForwardError",
    "attention",
    "causal_mask",
    "expert_geglu",
    "gelu",
    "moe_combine",
    "resolve_roles",
    "rmsnorm",
    "rope",
    "rotate_half",
    "router_logits",
    "top_k_experts",
]

# --- Architecture constants (xai-org/grok-1 run.py, grok-1 ckpt-0) -----------
MODEL_SIZE = 6144
NUM_Q_HEADS = 48
NUM_KV_HEADS = 8
KEY_SIZE = 128
NUM_EXPERTS = 8
NUM_SELECTED_EXPERTS = 2
FFN_SIZE = 32768
# 1/sqrt(128); upstream hardcodes the decimal expansion.
ATTN_OUTPUT_MULTIPLIER = 0.08838834764831845
MAX_ATTN_VAL = 30.0
ROPE_BASE = 10000
RMS_EPS = 1e-5
# ``Transformer.__call__`` does ``input_embeddings *= embedding_multiplier_scale``
# before the first block (model.py:1237). This is NOT optional for a residual
# forward: RMSNorm makes the *pre-attention* path invariant to a global input
# scale, but ``h = h0 + rmsnorm(attn_out)`` compares the two magnitudes directly.
# Grok-1's raw embedding rows have RMS ~0.01274, and 0.01274 * 78.3837 ~= 0.9989,
# so the multiplier is what puts the residual stream at unit RMS -- commensurate
# with the post-norm sublayer deltas. Omitting it inflates the attention delta by
# ~78x relative to the stream and destroys the residual's dilution of damage.
EMBEDDING_MULTIPLIER = 78.38367176906169


class ForwardError(RuntimeError):
    """Weight set does not match the Grok-1 block contract."""


@dataclass(frozen=True)
class HeadConfig:
    """Grouped-query head geometry.

    Parameterized only so tests can exercise the attention einsums at tiny
    dimensions; :data:`GROK1_HEADS` is the real checkpoint configuration.
    """

    n_q_heads: int = NUM_Q_HEADS
    n_kv_heads: int = NUM_KV_HEADS
    head_dim: int = KEY_SIZE

    def __post_init__(self) -> None:
        if self.n_q_heads % self.n_kv_heads:
            raise ForwardError(
                f"{self.n_q_heads} query heads is not a multiple of "
                f"{self.n_kv_heads} kv heads"
            )

    @property
    def groups(self) -> int:
        """Query heads sharing each key/value head."""
        return self.n_q_heads // self.n_kv_heads


GROK1_HEADS = HeadConfig()


# --- Slot -> architectural role ---------------------------------------------
# ckpt-0 flattens parameters in ALPHABETICAL order of their Haiku parameter
# path, not module creation order. Within one DecoderLayer the paths sort as
# ``moe_layer/... < multi_head_attention/... < rms_norm{,_1,_2,_3} < router``,
# which reproduces observed slots 00..11 exactly. Five independent checks agree:
#
# 1. MoELayer._inference_call indexes params["linear"], params["linear_1"] and
#    params["linear_v"] by name, and sorted those are gate/down/up -- matching
#    the observed (6144,32768), (32768,6144), (6144,32768) shapes.
# 2. MultiHeadAttention names "query"/"key"/"value" but leaves the output
#    projection unnamed, so Haiku calls it "linear"; sorted key < linear <
#    query < value gives narrow/wide/wide/narrow == slots 03..06. Creation
#    order would instead give wide/narrow/narrow/wide, which does not match.
# 3. The four RMSNorm scales sort rms_norm < rms_norm_1 < rms_norm_2 <
#    rms_norm_3, which here coincides with DecoderLayer call order.
# 4. Tensor-parallel grouped-scale counts from the checkpoint: only slot_04 and
#    slot_01 carry groups=8, i.e. a sharded *contracting* axis -- the signature
#    of an output/down projection. Q/K/V and gate/up shard their output axis
#    (groups=1). This is physical checkpoint evidence, independent of naming.
# 5. Shapes are self-consistent: 48*128 == 6144 for query/output, 8*128 == 1024
#    for key/value, ffn_size(6144, 8) == 32768.
SLOT_ROLES: dict[str, str] = {
    "slot_00": "expert_gelu",  # Haiku "linear"   -- GELU branch of the GeGLU
    "slot_01": "expert_down",  # Haiku "linear_1" -- down projection
    "slot_02": "expert_value",  # Haiku "linear_v" -- un-activated branch
    "slot_03": "key",
    "slot_04": "attn_out",  # Haiku "linear" -- output projection
    "slot_05": "query",
    "slot_06": "value",
    "slot_07": "norm_pre_attn",
    "slot_08": "norm_post_attn",
    "slot_09": "norm_pre_moe",  # router input
    "slot_10": "norm_post_moe",
    "slot_11": "router",
}

# Expected shape per role, with ``None`` for the leading expert axis.
_ROLE_SHAPES: dict[str, tuple[int, ...]] = {
    "expert_gelu": (NUM_EXPERTS, MODEL_SIZE, FFN_SIZE),
    "expert_down": (NUM_EXPERTS, FFN_SIZE, MODEL_SIZE),
    "expert_value": (NUM_EXPERTS, MODEL_SIZE, FFN_SIZE),
    "key": (MODEL_SIZE, NUM_KV_HEADS * KEY_SIZE),
    "attn_out": (MODEL_SIZE, MODEL_SIZE),
    "query": (MODEL_SIZE, NUM_Q_HEADS * KEY_SIZE),
    "value": (MODEL_SIZE, NUM_KV_HEADS * KEY_SIZE),
    "norm_pre_attn": (MODEL_SIZE,),
    "norm_post_attn": (MODEL_SIZE,),
    "norm_pre_moe": (MODEL_SIZE,),
    "norm_post_moe": (MODEL_SIZE,),
    "router": (MODEL_SIZE, NUM_EXPERTS),
}


def _slot_of(structural_name: str) -> str:
    """Extract the ``slot_NN`` component of a structural tensor name."""
    for part in structural_name.split("."):
        if part.startswith("slot_"):
            return part
    raise ForwardError(f"{structural_name}: no slot_NN component in structural name")


def resolve_roles(
    shapes: dict[str, tuple[int, ...]], *, require_complete: bool = True
) -> dict[str, str]:
    """Map structural tensor names to architectural roles, verifying shapes.

    ``shapes`` maps structural name -> tensor shape. Returns ``{role: name}``.
    Raises :class:`ForwardError` when a slot is unknown, duplicated, missing, or
    carries a shape the role cannot have -- the mapping is load-bearing for every
    downstream metric, so a mismatch must abort rather than be reinterpreted.

    ``require_complete=False`` permits a partial tier (an attention-only pack, for
    example), for callers that supply the remaining roles from another source.
    """
    by_role: dict[str, str] = {}
    for name, shape in shapes.items():
        slot = _slot_of(name)
        role = SLOT_ROLES.get(slot)
        if role is None:
            raise ForwardError(f"{name}: unknown slot {slot!r} for a Grok-1 block")
        if role in by_role:
            raise ForwardError(f"{role}: claimed by both {by_role[role]} and {name}")
        expected = _ROLE_SHAPES[role]
        if tuple(int(d) for d in shape) != expected:
            raise ForwardError(
                f"{name}: role {role} expects shape {expected}, got {tuple(shape)} -- "
                "slot/role mapping or checkpoint layout has changed"
            )
        by_role[role] = name
    missing = sorted(set(_ROLE_SHAPES) - set(by_role))
    if missing and require_complete:
        raise ForwardError(f"block is missing required tensors for roles: {missing}")
    return by_role


# --- Normalization ----------------------------------------------------------
def rmsnorm(x: np.ndarray, gain: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    """RMSNorm exactly as ``RMSNorm.__call__``: ``gain * x * rsqrt(ms + eps)``.

    Upstream computes in float32 and applies the learned scale as a plain
    multiply -- the scale is initialized to ``Constant(0)`` but is *not* used as
    ``1 + scale``. Because ``rmsnorm(c*x) == rmsnorm(x)`` for ``c > 0``, results
    are invariant to Grok-1's constant embedding multiplier.
    """
    x32 = np.asarray(x, dtype=np.float32)
    ms = np.mean(np.square(x32), axis=-1, keepdims=True, dtype=np.float32)
    return (x32 / np.sqrt(ms + np.float32(eps))) * np.asarray(gain, dtype=np.float32)


# --- Rotary embeddings ------------------------------------------------------
def rotate_half(x: np.ndarray) -> np.ndarray:
    """``concat(-x2, x1)`` over the split halves of the last axis."""
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate((-x2, x1), axis=-1)


def rope(x: np.ndarray, offset: int = 0, base: int = ROPE_BASE) -> np.ndarray:
    """Apply RoPE to ``(T, H, D)`` heads, matching ``RotaryEmbedding.__call__``.

    Half-split (GPT-NeoX) convention: the phase for the 64 inverse frequencies
    is tiled twice across the 128-wide head, so dimension ``d`` and ``d + D/2``
    share a phase. Upstream applies this to query and key heads only.
    """
    dim = x.shape[-1]
    if dim % 2:
        raise ForwardError(f"RoPE needs an even head dim, got {dim}")
    exponents = np.arange(0, dim, 2, dtype=np.float32)
    inv_freq = (1.0 / (np.float32(base) ** (exponents / np.float32(dim)))).astype(np.float32)
    t = np.arange(x.shape[0], dtype=np.float32) + np.float32(offset)
    phase = np.tile(np.einsum("i,j->ij", t, inv_freq), (1, 2))[:, None, :]
    return x * np.cos(phase).astype(np.float32) + rotate_half(x) * np.sin(phase).astype(np.float32)


# --- Attention --------------------------------------------------------------
def causal_mask(tokens: int) -> np.ndarray:
    """Lower-triangular self-attention mask of shape ``(T, T)``."""
    return np.tril(np.ones((tokens, tokens), dtype=bool))


def _softmax_fp32(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis in float32, as upstream always does."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return exp / exp.sum(axis=-1, keepdims=True, dtype=np.float32)


def attention(
    h_normed: np.ndarray,
    w_query: np.ndarray,
    w_key: np.ndarray,
    w_value: np.ndarray,
    w_out: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    offset: int = 0,
    heads: HeadConfig = GROK1_HEADS,
) -> np.ndarray:
    """Grouped-query self-attention over ``(T, MODEL_SIZE)`` normed activations.

    All Grok-1 ``Linear`` params are stored ``[in_size, out_size]``, so every
    projection is a plain ``x @ w`` with no transpose.
    """
    tokens = h_normed.shape[0]
    n_q_heads, n_kv_heads, head_dim = heads.n_q_heads, heads.n_kv_heads, heads.head_dim
    q = (h_normed @ w_query).reshape(tokens, n_q_heads, head_dim)
    k = (h_normed @ w_key).reshape(tokens, n_kv_heads, head_dim)
    v = (h_normed @ w_value).reshape(tokens, n_kv_heads, head_dim)
    # RoPE is applied to query and key only, never to value.
    q = rope(q, offset=offset)
    k = rope(k, offset=offset)
    # Grouped query heads: q head ``g*groups + j`` attends with kv head ``g``.
    # Upstream does reshape(b, t, kv_h, h // kv_h, d), i.e. contiguous grouping.
    qg = q.reshape(tokens, n_kv_heads, heads.groups, head_dim)

    logits = np.einsum("thHd,Thd->hHtT", qg, k, dtype=np.float32)
    logits *= np.float32(ATTN_OUTPUT_MULTIPLIER)
    # tanh soft-cap: bounds logits to +/-30 before masking.
    cap = np.float32(MAX_ATTN_VAL)
    logits = cap * np.tanh(logits / cap)
    if mask is None:
        mask = causal_mask(tokens)
    logits = np.where(mask[None, None, :, :], logits, np.float32(-1e30))

    weights = _softmax_fp32(logits)
    ctx = np.einsum("hHtT,Thd->thHd", weights, v, dtype=np.float32)
    return ctx.reshape(tokens, n_q_heads * head_dim) @ w_out


# --- Mixture of experts -----------------------------------------------------
_GELU_COEFF = np.float32(0.044715)
_SQRT_2_OVER_PI = np.float32(0.7978845608028654)  # sqrt(2/pi)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU in the tanh formulation used by ``jax.nn.gelu`` **by default**.

    ``jax.nn.gelu(x, approximate=True)`` is the JAX signature -- unlike PyTorch,
    the approximate form is the default -- and upstream calls it with defaults.
    So the faithful kernel is
    ``0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))``, not the erf form. The two
    differ by up to ~1e-3 absolute, which is not negligible against ternary
    routing margins, so this must match upstream rather than be "more exact".
    """
    x32 = np.asarray(x, dtype=np.float32)
    inner = _SQRT_2_OVER_PI * (x32 + _GELU_COEFF * x32 * x32 * x32)
    return (np.float32(0.5) * x32 * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)


def router_logits(h_normed: np.ndarray, w_router: np.ndarray) -> np.ndarray:
    """Routing logits ``h @ w`` in float32, as ``Router._router_weights`` does."""
    return np.asarray(h_normed, dtype=np.float32) @ np.asarray(w_router, dtype=np.float32)


def top_k_experts(
    logits: np.ndarray, k: int = NUM_SELECTED_EXPERTS
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, gates)`` for the top-``k`` experts per token.

    Upstream takes ``top_k`` of the softmax over **all** experts and uses those
    raw probabilities as combination weights -- they are deliberately *not*
    renormalized over the selected subset, so the gates sum to less than one.
    Softmax is monotone, so selection by probability equals selection by logit.
    """
    probs = _softmax_fp32(np.asarray(logits, dtype=np.float32))
    # Negate for a descending sort; argpartition alone does not order the top-k.
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
    gates = np.take_along_axis(probs, idx, axis=-1)
    return idx.astype(np.int64), gates.astype(np.float32)


def expert_geglu(
    h_rows: np.ndarray, w_gelu: np.ndarray, w_value: np.ndarray, w_down: np.ndarray
) -> np.ndarray:
    """One expert's GeGLU feed-forward: ``down(gelu(gelu_branch) * value_branch)``.

    Mirrors ``DenseBlock`` together with ``MoELayer._inference_call``: the GELU
    lands on the ``linear`` branch (slot_00, labelled ``gate`` by xai-dissect),
    while ``linear_v`` (slot_02, labelled ``up``) passes through un-activated.
    Swapping the two is shape-legal and therefore a silent correctness risk.
    """
    activated = gelu(h_rows @ w_gelu)
    passthrough = np.asarray(h_rows @ w_value, dtype=np.float32)
    return (activated * passthrough) @ w_down


def moe_combine(
    outputs: dict[int, np.ndarray], idx: np.ndarray, gates: np.ndarray, tokens: int
) -> np.ndarray:
    """Weighted sum of per-expert outputs using the raw softmax gates.

    ``outputs`` maps expert index -> that expert's output for the tokens routed
    to it, in the order those tokens appear in ``idx``.
    """
    width = next(iter(outputs.values())).shape[1] if outputs else MODEL_SIZE
    out = np.zeros((tokens, width), dtype=np.float32)
    for expert, rows in outputs.items():
        hits = np.argwhere(idx == expert)
        if hits.shape[0] != rows.shape[0]:
            raise ForwardError(
                f"expert {expert}: {hits.shape[0]} routed tokens but {rows.shape[0]} outputs"
            )
        token_ids, slots = hits[:, 0], hits[:, 1]
        out[token_ids] += gates[token_ids, slots][:, None] * rows
    return out
