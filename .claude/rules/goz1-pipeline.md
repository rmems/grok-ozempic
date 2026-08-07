# GOZ1 pipeline invariants

## Two CLI paths (do not confuse them)

| Path | Commands | Reads weights? |
|------|----------|----------------|
| **SAAQ metadata** | `validate-ingest`, `smoke-grok1`, `convert-grok1`, `validate-grok1-artifact` | Index/metadata only (may hash shards if checksums present) |
| **GOZ1 real quant** | `quantize-goz1` | Yes — **safetensors or `.npy` only** |

Official Grok-1 `ckpt-0` shards are JAX **pickle**. `quantize-goz1` / `run_quantization` **reject pickle**.

### Export script scope

Two exporters, different jobs — pick by source dtype.

`scripts/export_grok1_embedding_npy.py` is **stdlib-only** and **f32-only**. Defaults target the token embedding on one pickle shard (not a full 770-tensor pack). With `--stem` and layout flags it can export **another single f32 tensor** from a shard — still one file per invocation, not bulk export.

```bash
python3 scripts/export_grok1_embedding_npy.py \
  --shard "$CKPT/tensor00000_000" \
  --output-dir "$OUT"
```

`scripts/export_grok1_int8_npy.py` (**requires numpy**) handles everything the f32 exporter cannot: official ckpt-0 ships all attention projections and MoE experts as `__main__.QuantizedWeight8bit` (int8 `weight` × bfloat16 `scales`), which the float stream rejects. The **pilot contract** is manifest-driven whole-block export via `--block`/`--mode` (quantized *and* f32 preserve tiers in one invocation). `scripts/block_pilot_goz1.sh` uses only that path.

```bash
python3 scripts/export_grok1_int8_npy.py \
  --conversion-manifest "$RUN3/conversion-manifest.json" \
  --block 0 --mode attention_plus_expert \
  --output-dir "$OUT"
```

`--structural-name` (repeatable; may omit `--block`) is a **debug/repair hatch** for re-exporting individual tensors — not part of the pilot contract. Partial exports are fine for inspection or fixing a single bad npy. Do not treat `--structural-name` as the supported production export mode.

⚠ **V2 fail-closed does not detect under-packing.** It rejects an *input name that matches no rule* — a misclassification guard. A tensor that is simply **absent** from the npy directory produces no name to match, so V2 is silent about it and the pack comes out short. Completeness is enforced separately, by explicit inventory and counter checks against the xai-dissect **conversion manifest**:

| Check | Where |
|-------|-------|
| Expected ternary / preserve names per block+mode vs pack contents | `scripts/route_preservation_metrics.py` (`_validate_ternary_inventory`, `_validate_preserve_inventory`) |
| Pack ternary / preserve counters vs the selected mode | `scripts/block_pilot_goz1.sh` step 4b |

Without `--conversion-manifest` the metrics run is **diagnostic-only**: it prints observed values but reports thresholded rows as `diagnostic`, records `certification.certified: false`, and exits non-zero. A kinds-only check cannot certify a gate, because matching the *kinds present* says nothing about whether every expected tensor is there.

Scales are **grouped along the contracting axis**: weight `(*lead, K, N)`, scales `(*lead, G, N)`, `K % G == 0`, so `w_f32 = weight.reshape(*lead, G, K//G, N) * scales[..., :, None, :]`. `G` is the tensor-parallel shard count of that axis (8 when `K` was sharded, 1 when `N` was). Verified bit-exact against unpickled ground truth — see `reports/grok-1-block-pilot/results.md`.

Both use stem mapping `embedding__slot_00__token_embedding.npy` → logical `embedding.slot_00.token_embedding` (`__` → `.`).

### Bounded block pilot

`scripts/block_pilot_goz1.sh` runs the whole #53 loop: dequant export → tier-aware V2 manifest derived at runtime (never mutating `dissect/`) → `quantize-goz1 --verify` → exact trit histogram → route-preservation metrics.

```bash
BLOCK=0 MODE=attention_plus_expert scripts/block_pilot_goz1.sh
```

The deriver hard-fails if a preserve rule names a family outside run3's `keep_fp32`, or a ternary candidate names one outside `pilot_quantize`.

## Real pack recipes

**NPY directory** (JAX export path; stems are structural, so prefer the V2 structural manifest):

```bash
cargo run --release --features cli --locked -- quantize-goz1 \
  --input-dir /path/to/npy-dir \
  --output /path/to/out.goz1 \
  --manifest dissect/grok-1/structural-manifest.json \
  --input-format npy \
  --verify
```

**Safetensors — structural names** (V2):

```bash
cargo run --release --features cli --locked -- quantize-goz1 \
  --input-dir /path/to/safetensors-dir \
  --output /path/to/out.goz1 \
  --manifest dissect/grok-1/structural-manifest.json \
  --input-format safetensors \
  --verify
```

**Safetensors — legacy `blk.*` names** (V1; do not pair with V2):

```bash
cargo run --release --features cli --locked -- quantize-goz1 \
  --input-dir /path/to/safetensors-dir \
  --output /path/to/out.goz1 \
  --manifest dissect/grok-1/baseline.json \
  --input-format safetensors \
  --verify
```

Since #40 / RM-191, **runtime** packs prefer the V2 `structural-manifest.json` whenever tensor names are structural (`block_{NNN}.slot_{SS}.{kind}`, export-script stems). V2 is fail-closed: an unmatched name hard-errors instead of defaulting to ternary. For legacy `blk.*`-named inputs use V1 `baseline.json` (defaults fallthrough allowed). Prefer `--verify`. Authoritative structural names / planning surface: `~/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN/manifests/xai-grok-1-ckpt-0/` (tests honor `GROK_OZEMPIC_DISSECT_RUN`).

**Evidence sources:**

| Metric | Where |
|--------|--------|
| Ternary / fp16-preserve counts | Pack CLI summary line (`… X ternary, Y fp16/preserve`) |
| Total GOZ1 **file_size** (bytes) | `--verify` line: `GOZ1 verify ok: … file_size=…` |
| Wall / max RSS | External: `/usr/bin/time -v`, `gtime -v`, or BSD `time -l` |
| Exact trit counts / sparsity | `scripts/goz1_trit_histogram.py PACK.goz1` |
| Per-tensor reconstruction scale | Same histogram (`scale` field); `None` means a legacy v1 pack |
| Container version | `--verify` line: `GOZ1 verify ok: version=…` |
| Route-preservation gates | `scripts/route_preservation_metrics.py` (fills run3's `unknown` surface) |

Claim ternary only when the CLI ternary counter matches expectation.

## Container versions (#65)

Packs are **GOZ1 v2**: each tensor row carries the reconstruction-optimal scale
`α*`, so a pack dequantizes from its own contents (`value = scale × payload`).
v1 packs have no scale field and are still readable; consumers fall back to the
oracle α derived from the source npy **only** there, and must tag it
(`PackWeights.scale_sources` → `legacy_oracle` vs `pack_v2`).

An oracle figure is a lower bound no runtime can reproduce, so never report one
as a pack-only measurement. Full layout and policy: `docs/goz1-format.md`.

⚠ **`oz.gif_threshold` is never authoritative — it records `defaults || config` only.** A manifest carrying per-tensor `ternary_candidates[].gif_threshold` still reports the baseline, which under a tier-aware manifest can be a value *no tensor in the pack used* (#51 / #58 trap; demonstrated in `reports/grok-1-block-pilot/results.md`).

**Fixed for v3 packs (#66):** each tensor row carries the applied τ, so read it from there.

| Pack | How to get the applied τ |
|------|--------------------------|
| **v3** | Tensor row: `gif_threshold` (multiplier) and `threshold_abs` (the cut compared against `\|w\|`). `--verify` prints the distinct values; `goz1_trit_histogram.py` reports both per tensor. `rms = threshold_abs / gif_threshold` |
| **v1 / v2** | Not recorded. Infer from measured sparsity — the metadata key cannot tell you |

v3 packs also carry `oz.gif_threshold_authority = "tensor_row"` and
`oz.gif_threshold_scope = "baseline_only_not_applied"`, so the pack says of itself
that the scalar key is not the applied value.

## Ownership

- GOZ1 format, streaming, manifests, Grok-1 glue → this crate.
- CUDA / ternary GEMV kernels → `myelin-accelerator` (see `kernel-boundary` rule).
