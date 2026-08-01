# /quantize-embed — first real embedding → GOZ1

Reference experiment: GH **#39** / Linear **RM-190** (results under `reports/grok-1-first-embed-goz1/` when present).

## Cloud warning

Claude Code **on the web** does **not** have `~/.models/xai-grok-1`. Only run the weight steps on a machine (or Remote Control session) that has the checkpoint. On pure cloud VMs, implement code/tests only.

## 1. Export pickle → `.npy`

```bash
CKPT="${CKPT:-$HOME/.models/xai-grok-1/ckpt-0}"
OUT="${OUT:-$HOME/.models/xai-grok-1/export-npy}"
mkdir -p "$OUT"
python3 scripts/export_grok1_embedding_npy.py \
  --shard "$CKPT/tensor00000_000" \
  --output-dir "$OUT"
```

Expect `embedding__slot_00__token_embedding.npy` (~3.1 GiB f32, shape `131072×6144`).

## 2. Pack GOZ1 (V1 baseline until #40)

```bash
ART="${ART:-$HOME/.models/xai-grok-1/artifacts}"
mkdir -p "$ART"
cargo run --release --features cli -- quantize-goz1 \
  --input-dir "$OUT" \
  --output "$ART/grok1-first-embed.goz1" \
  --manifest dissect/grok-1/baseline.json \
  --input-format npy \
  --verify
```

## 3. Accept only if

- Output exists and is non-trivial (~192 MiB ternary for single embedding historically)
- CLI summary shows **ternary** path (not preserve-only)
- Record wall time, max RSS, command line

Do **not** use `structural-manifest.json` for runtime pack until #40 is done.
