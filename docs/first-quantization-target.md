# First Quantization Target for `grok-ozempic`

This document defines the first bounded quantization target for [`grok-ozempic`](../README.md) under GitHub issue `#14` / Linear `MET-76`. It is now implemented as part of the combined Grok-1 SAAQ artifact flow documented in [`grok1-saaq-artifact-flow.md`](./grok1-saaq-artifact-flow.md).

## Goal

Select one concrete first quantization target that is:

- explicit enough to anchor validation and artifact expectations
- small enough to land in one implementation PR
- compatible with the existing GOZ1 export contract

## Selected target

The first quantization target is:

- `embedding.slot_00.token_embedding`

This target already appears as the clearest candidate in [`reports/grok-1-official__ckpt-0/saaq-readiness.md`](../reports/grok-1-official__ckpt-0/saaq-readiness.md).

## Why this target

`embedding.slot_00.token_embedding` is the best first target because it is:

- a single bounded tensor family rather than a repeated expert family
- easier to validate than a multi-block rollout
- not part of the routing-critical path described by the existing artifact reports
- compatible with the current ternary quantization flow implemented in [`src/core/quantizer.rs`](../src/core/quantizer.rs)

## Expected input artifacts

The first implementation PR should assume the following inputs:

1. One xai-dissect manifest JSON.
   - Resolution precedence is already documented in [`docs/dissect-manifest.md`](./dissect-manifest.md).
   - Runtime resolution is implemented in [`resolve_manifest()`](../src/core/stream.rs:108).

2. One supported checkpoint source.
   - Supported input layouts are defined by [`QuantizationInputFormat`](../src/types.rs:40).
   - The checkpoint source may be safetensors shards or a flat `.npy` directory.

3. A concrete mapping from the logical target name to the physical checkpoint tensor name.
   - This mapping is still a known unknown and must be confirmed during implementation.

## Expected output artifact

The expected output artifact for the first implementation PR is:

- one GOZ1 packed checkpoint file

The format must remain the existing GOZ1 container defined in [`src/core/weight_pack.rs`](../src/core/weight_pack.rs).

### Output contract

- The selected token-embedding tensor must be emitted through the ternary quantization path.
- Ternary packing must continue to use the existing encoding path in [`pack_trits()`](../src/core/quantizer.rs:62).
- Source tensors that are still preserve/FP16 must remain on their existing paths.
- Routing-critical tensors must remain unchanged.
- For the sprint metadata writer, the selected target is represented in `artifact.index.json` with `candidate_saaq_embedding` policy under the `saaq-g1-v0` custom-format contract. The older GOZ1 streaming path remains separate and unchanged.

## Explicit non-goals for this issue

This issue does **not** do any of the following:

- broaden the first target into multiple quantization targets
- quantize routing tensors
- quantize the full expert tensor surface
- change SAAQ math
- change router math
- change dequantization behavior
- introduce cloud execution
- mix runtime integration or research narrative work into the same PR
- change heartbeat behavior, because heartbeat behavior is no longer part of this project

## Assumptions

- The readiness report is currently the best repository-local signal for choosing the first target.
- GOZ1 is the intended export artifact for quantized weights.
- Manifest-driven targeting is preferred over legacy substring fallback heuristics.

## Constraints

- The first implementation must stay small enough for one PR.
- The work must preserve existing GOZ1 writer semantics.
- The first target must be specific enough for downstream validation work.
- Routing-critical tensors remain excluded from the first quantized target set.

## Known unknowns

- The exact physical checkpoint tensor name for `embedding.slot_00.token_embedding` is not yet documented.
- The source dtype for the target tensor still needs confirmation from real checkpoint data.
- The current fallback manifest in [`dissect/grok-1/baseline.json`](../dissect/grok-1/baseline.json) may need an explicit token-embedding rule if upstream manifest coverage is insufficient.
- A dedicated downstream validation contract for the one-target GOZ1 case still needs to be written in implementation work.

## Implementation boundary for the first PR

The follow-up implementation PR should be limited to:

1. confirming the physical checkpoint mapping for `embedding.slot_00.token_embedding`
2. encoding that target as the first manifest-driven ternary quantization target
3. producing one normal GOZ1 artifact with the token embedding quantized
4. preserving current handling for routing-critical tensors
5. adding only the minimum validation needed to prove the target flows through the existing GOZ1 export path

## Recommended branch

- `feat/met-76-first-quant-target-token-embedding`

## Attribution

Planning decision prepared by Roo Code agent openAI/GPT-5.4.
