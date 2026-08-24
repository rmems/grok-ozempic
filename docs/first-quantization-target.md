# Verified first quantization target for `grok-ozempic`

This document records the observed contract for the first real Grok-1 weight
experiment. It replaces the pre-implementation plan from GitHub #14 / Linear
MET-76. For the copyable end-to-end commands, see the
[Grok-1 SAAQ artifact flow](./grok1-saaq-artifact-flow.md).

## Physical-to-logical mapping

The official `xai-org/grok-1` `ckpt-0` token embedding is known and measured:

| Layer | Value |
|---|---|
| Physical checkpoint shard | `tensor00000_000` |
| Pickle payload offset | `151` (`0x97`) |
| Source dtype and shape | f32 `(131072, 6144)` |
| Exported NPY filename | `embedding__slot_00__token_embedding.npy` |
| Runtime logical name | `embedding.slot_00.token_embedding` |

The exporter uses `__` in the filename stem because the NPY loader maps `__`
back to `.`. The mapping is therefore structural and deterministic; it is not a
substring guess.

## Why an export is required

Official Grok-1 checkpoint shards are JAX pickle frames. The real-weight
`quantize-goz1` path accepts safetensors or a flat NPY directory, not pickle.
Run [`scripts/export_grok1_embedding_npy.py`](../scripts/export_grok1_embedding_npy.py)
before packing the official embedding shard. The exporter validates the known
offset, byte count, dtype, and shape, and writes a stream-compatible `.npy`.

This requirement is independent of the metadata-only SAAQ commands. The two
pipelines have different outputs:

| Pipeline | Purpose | Weight payloads |
|---|---|---|
| `validate-ingest` → `smoke-grok1` → `convert-grok1` → `validate-grok1-artifact` | Validate the `saaq-g1-v0` plan and deterministic structural indexes | Does not quantize or pack weights; `validate-ingest` may hash files named by `checksums.json` |
| pickle export → `quantize-goz1 --verify` | Export and quantize real tensor values into a GOZ1 container | Reads the 3 GiB f32 embedding payload and writes packed weights |

An `artifact.index.json` from the first pipeline is not a GOZ1 checkpoint.

## Manifest and safety contract

Use a manifest that explicitly covers the structural NPY name. Runtime support
for V2 structural names landed in PR #55: under a V2 manifest, an unmatched
tensor fails closed instead of falling through to defaults. The in-tree
[`dissect/grok-1/structural-manifest.json`](../dissect/grok-1/structural-manifest.json)
is a non-authoritative policy reference; the latest correct xai-dissect run is
the authoritative planning surface. See
[`dissect-manifest.md`](./dissect-manifest.md) for resolution precedence.

Routers and norms remain protected. This one-tensor experiment does not test
full-model inference, routing preservation, or downstream model quality, and it
does not justify expanding ternary selection to routing-critical tensors.

## Observed first artifact

The historical first pack used the baseline manifest and GOZ1 version 1. It
produced one verified `201327136`-byte artifact (about 192.00 MiB), compressing
the 3 GiB f32 payload by about 16×. Its trit distribution at `gif_threshold =
0.05` was about 4.17% zero, 47.7% positive, and 48.2% negative, so the result was
dense/sign-like rather than strongly sparse.

The immutable measurements and exact historical command are in
[`reports/grok-1-first-embed-goz1/results.md`](../reports/grok-1-first-embed-goz1/results.md).
New packs use the current GOZ1 version 3 writer, which adds per-tensor
reconstruction scale and applied-threshold fields; see
[`goz1-format.md`](./goz1-format.md). Do not compare the old artifact's version
number with a newly generated file as though they were byte-identical formats.

## Remaining validation boundaries

The physical name and source dtype are no longer unknown. The outstanding
boundaries are deliberately tracked elsewhere:

- GitHub #35 validates the complete local checkpoint inventory and size.
- GitHub #36 exercises the full 770-tensor / 64-router metadata gate.
- GitHub #85 and PR #89 contain the later supervised multi-block precision
  experiment; its INT4 research side tables are not GOZ1 containers.

The first embedding result is evidence that one real tensor can be exported,
packed, and verified. It is not evidence that a complete model is usable or
that the broader ingest gates are complete.

## Scope preserved by this contract

- Do not quantize routers or norms.
- Do not change quantization, routing, dequantization, or GOZ1 layout semantics
  as part of this documentation contract.
- Do not treat pickle as a direct `quantize-goz1` input.
- Do not treat SAAQ metadata conversion as real-weight quantization.
