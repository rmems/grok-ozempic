# GOZ1 container format

`GOZ1` is grok-ozempic's packed-checkpoint container (**not** GGUF). Little-endian
throughout. The authoritative implementation is `src/core/weight_pack.rs`
(writer) and `src/core/weight_pack_read.rs` (reader/verifier); the stdlib parser
in `scripts/goz1_trit_histogram.py` mirrors them and must be kept in step.

## Layout

```
magic         u32     ASCII "GOZ1"  (format family — does not change per version)
version       u32     layout version (see below)
tensor_count  u64
meta_count    u64

metadata[meta_count]:
  key         u64 len + utf8
  vtype       u32     0 = u32, 1 = str
  value       u32 | (u64 len + utf8)

tensor_table[tensor_count]:
  name        u64 len + utf8
  ndim        u32
  shape       u64 × ndim        row-major, slowest index first
  tensor_type u32               0 = f16, 1 = ternary
  data_offset u64               relative to the data section
  scale       f32               ** version 2 only **
  sentinel    u32               ** version 2 only ** — 0x5CA1E021

<padding to DATA_ALIGNMENT = 32>
payload blobs, each padded to 32
```

The v2 sentinel closes the row. Because `scale` is patched at `finalize` rather
than written in place, a row whose scale was never supplied is otherwise
indistinguishable from a well-formed one at parse time; a fixed trailing value
turns any desync between writer and reader — a miscounted field, or a row width
changed on one side only — into an immediate error rather than a plausible
misparse of the following row. Readers must reject a row whose sentinel does not
match (`OZ1_V2_ROW_SENTINEL` in Rust; the same constant in the Python parser).

`data_offset` and `scale` are both written as placeholders and patched in
`finalize`, because the whole tensor table is laid down before any payload is
quantized — neither value is known when its row is written.

## Versions

| Version | Tensor row | Status |
|---------|------------|--------|
| 1 | ends at `data_offset` | **legacy, read-only.** Still parsed; never written by current builds |
| 2 | appends `scale: f32` + `sentinel: u32` | **current.** Written by `quantize-goz1` / `run_quantization` |
| other | — | **rejected** |

The v2 row is a strict *append*, so a reader parses the common prefix and reads
the scale only when the version says it is there.

Unknown versions are refused rather than parsed as a compatible prefix: a future
layout could reorder the row, and silently misreading offsets would yield a
plausible-looking report for a pack we do not actually understand.

The binary `version` field is what gates the layout. The `oz.quantization_version`
metadata key carries the same number for anything reading metadata only; both
move together.

## Why version 2 exists (GH #65)

A ternary payload is 2 bits per weight and nothing else — it carries **sign but
no magnitude**. Under v1 the container persisted neither the quantizer's `rms`
nor its threshold, so `w ≈ α·t` had no recoverable `α` and **a pack could not be
dequantized from its own contents**. Every consumer had to re-derive a scale from
the original checkpoint, which meant a pack was not a self-contained artifact:
the route-preservation figures in #61 / PR #64 were all produced with an *oracle*
α computed from the source weights, a number no runtime could obtain.

Version 2 stores that scale per tensor.

### Which scale

The **reconstruction-optimal** single scale:

```
α* = Σ(w·t) / count(t ≠ 0)        over fired positions
```

This minimizes `‖w − α·t‖²` for the pack's own trit pattern (`Σt² = count(fired)`,
since every fired trit is ±1), so it is the best single scale any consumer could
pick — persisting it costs nothing against the format's ceiling.

Note this is *not* `rms` or `threshold`, which is what `quantize_f32` already
computed for gating. It is a different quantity, computed in the same pass.

The numerator is the **signed** product, not `Σ|w|`. The two agree by
construction — the GIF gate assigns `+1` only above `+τ` and `-1` only below `−τ`,
so `sign(t) == sign(w)` wherever a trit fires — but the signed form is the actual
least-squares numerator and degrades honestly (a smaller α) rather than inflating
if that invariant is ever broken.

### Granularity

**Per-tensor**, matching how `quantize_f32` works. Per-output-channel was measured
in #52 / #57 at ≈0.896 cosine versus ≈0.887 per-tensor — a real but small gain for
a much larger header. The field does not preclude a later per-channel variant,
which would be a further version bump.

### Edge cases

| Case | Stored | Why |
|------|--------|-----|
| Nothing fires (fully sparse tensor) | `0.0` | α is undefined with no fired positions, and `α·t` is identically zero for any α, so nothing is lost. Must stay **finite** or the pack fails verification |
| Empty tensor | `0.0` | same |
| `TENSOR_F16` payload | `1.0` | the stored halves *are* the values, so `value = scale × payload` holds uniformly across both payload kinds |

Scales are validated per payload kind, at both write and verify time:

| Payload | Rejected |
|---------|----------|
| `TENSOR_TERNARY` | non-finite, or negative — α is a magnitude; sign lives in the trit, so a negative α silently inverts every weight in the tensor |
| `TENSOR_F16` | anything other than exactly `1.0` — the halves are the values, so any other scale means the row and the payload disagree |

`write_tensor_data` refuses these (naming the tensor while we still know it) and
`verify_pack_file` rejects them again on read, so a pack from any writer is
checked. The placeholder written during `begin` is `NaN` deliberately — a writer
that somehow finalized without supplying a real value produces a pack that fails
verification, rather than one that silently reconstructs every weight as zero.

## Consumer policy for v1 packs

v1 packs remain readable. Consumers must:

1. Prefer the stored scale whenever the pack has one.
2. Fall back to the oracle α **only** for v1, and record that fallback in report
   provenance — `scripts/grok1_block_weights.py` exposes
   `PackWeights.scale_sources`, which tags each tensor `pack_v2` or
   `legacy_oracle`.

The distinction is load-bearing, not cosmetic: an oracle figure is a *lower bound*
on quantization damage that no runtime can reproduce, so presenting one as a
pack-only measurement overstates what the format actually delivers.

In the Python reader, `entry["scale"] is None` is the operative signal — it means
the layout has no scale field at all, and is deliberately distinct from a stored
`0.0`, which is a legitimate value for a tensor where nothing fired.
`entry["container_version"]` carries the exact version for provenance.

## Reconstruction

Uniform across payload kinds:

```
value = scale × payload
```

where `payload` is the decoded trit (`0b00 = 0`, `0b01 = +1`, `0b10 = -1`,
four per byte, LSB-first) for `TENSOR_TERNARY`, and the stored half itself for
`TENSOR_F16`. A tensor's payload lives at `data_section_start + data_offset`;
`PackVerifyReport` exposes both so a caller can locate a blob from the report
alone.

## See also

- `.claude/rules/goz1-pipeline.md` — pipeline invariants and pack recipes
- `docs/ARCHITECTURE.md` — layer ownership
- GH #65 / RM-251 — the issue this version answers
