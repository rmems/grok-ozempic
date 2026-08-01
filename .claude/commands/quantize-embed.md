# /quantize-embed — first real embedding → GOZ1

Reference experiment: GH **#39** / Linear **RM-190** (results under `reports/grok-1-first-embed-goz1/` when present).

## Cloud warning

Claude Code **on the web** does **not** have `~/.models/xai-grok-1`. Only run the weight steps on a machine (or Remote Control session) that has the checkpoint. On pure cloud VMs, implement code/tests only.

## Timing helper

```bash
if command -v gtime >/dev/null 2>&1; then
  TIME_V=(gtime -v)
elif [ "$(uname -s)" = Darwin ]; then
  TIME_V=(/usr/bin/time -l)   # BSD time; RSS field differs from GNU -v
else
  TIME_V=(/usr/bin/time -v)
fi
```

## 1. Export pickle → isolated `.npy` dir (embedding only)

`export_grok1_embedding_npy.py` is **stdlib-only** and exports **one** tensor.

**Do not** wipe caller-owned export directories. Always write into a dedicated stage path:

```bash
CKPT="${CKPT:-$HOME/.models/xai-grok-1/ckpt-0}"
STAGE="${STAGE:-$HOME/.models/xai-grok-1/export-npy-embed-only-$$}"
mkdir -p "$STAGE"
"${TIME_V[@]}" python3 scripts/export_grok1_embedding_npy.py \
  --shard "$CKPT/tensor00000_000" \
  --output-dir "$STAGE"
```

Expect sole file `embedding__slot_00__token_embedding.npy` (f32 `131072×6144` → **3.0 GiB** payload + small NPY header).

```bash
# Guard: quantize-goz1 packs *every* .npy under --input-dir
mapfile -t NPYS < <(find "$STAGE" -maxdepth 1 -name '*.npy' | sort)
test "${#NPYS[@]}" -eq 1
test "$(basename "${NPYS[0]}")" = "embedding__slot_00__token_embedding.npy"
```

## 2. Pack GOZ1 (V1 baseline until #40)

```bash
ART="${ART:-$HOME/.models/xai-grok-1/artifacts}"
mkdir -p "$ART"
cargo build --release --features cli --locked
"${TIME_V[@]}" ./target/release/grok-ozempic quantize-goz1 \
  --input-dir "$STAGE" \
  --output "$ART/grok1-first-embed.goz1" \
  --manifest dissect/grok-1/baseline.json \
  --input-format npy \
  --verify
```

## 3. Accept only if

- Output exists and is non-trivial (~**192 MiB** ternary for single embedding historically) — measure size with `stat`/`ls`, not from CLI summary
- CLI summary shows **1 ternary** (and `fp16/preserve` as the combined non-ternary counter)
- Stage dir had only the embedding `.npy`
- Wall / max RSS from `TIME_V` (not from CLI summary)

Do **not** use `structural-manifest.json` for runtime pack until #40 is done.
