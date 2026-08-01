# /quantize-embed — first real embedding → GOZ1

Reference experiment: GH **#39** / Linear **RM-190** (results under `reports/grok-1-first-embed-goz1/` when present).

## Cloud warning

Claude Code **on the web** does **not** have `~/.models/xai-grok-1`. Only run the weight steps on a machine (or Remote Control session) that has the checkpoint. On pure cloud VMs, implement code/tests only.

## Timing helper

Prefer GNU `time -v` for wall + max RSS. On macOS without GNU time, install `gnu-time` (`gtime`) or use `/usr/bin/time -l` (RSS field differs).

```bash
TIME_V=(/usr/bin/time -v)
command -v gtime >/dev/null 2>&1 && TIME_V=(gtime -v)
```

## 1. Export pickle → `.npy` (embedding only)

`export_grok1_embedding_npy.py` is **stdlib-only** and exports **one** tensor.

```bash
CKPT="${CKPT:-$HOME/.models/xai-grok-1/ckpt-0}"
OUT="${OUT:-$HOME/.models/xai-grok-1/export-npy-embed-only}"
mkdir -p "$OUT"
# Isolate input: quantize-goz1 packs *every* .npy under --input-dir
rm -f "$OUT"/*.npy
"${TIME_V[@]}" python3 scripts/export_grok1_embedding_npy.py \
  --shard "$CKPT/tensor00000_000" \
  --output-dir "$OUT"
```

Expect sole file `embedding__slot_00__token_embedding.npy` (f32 `131072×6144` → **3.0 GiB** payload + small NPY header).

```bash
# optional guard
test "$(find "$OUT" -maxdepth 1 -name '*.npy' | wc -l)" -eq 1
```

## 2. Pack GOZ1 (V1 baseline until #40)

```bash
ART="${ART:-$HOME/.models/xai-grok-1/artifacts}"
mkdir -p "$ART"
cargo build --release --features cli --locked
"${TIME_V[@]}" ./target/release/grok-ozempic quantize-goz1 \
  --input-dir "$OUT" \
  --output "$ART/grok1-first-embed.goz1" \
  --manifest dissect/grok-1/baseline.json \
  --input-format npy \
  --verify
```

## 3. Accept only if

- Output exists and is non-trivial (~**192 MiB** ternary for single embedding historically)
- CLI summary shows **1 ternary** (and `fp16/preserve` as the combined non-ternary counter — not separate preserve vs fp16)
- Input dir had only the embedding `.npy`
- Record wall time, max RSS, and full command line

Do **not** use `structural-manifest.json` for runtime pack until #40 is done.
