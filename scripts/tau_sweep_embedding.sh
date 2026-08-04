#!/usr/bin/env bash
# τ (gif_threshold) sweep on the sole Grok-1 token embedding — GH #51 / RM-201.
#
# For each τ: quantize-goz1 with a manifest that has NO defaults.gif_threshold
# (so the CLI --gif-threshold actually controls the quantize τ — the stock
# dissect/grok-1/baseline.json bakes defaults.gif_threshold=0.05 which
# silently overrides the CLI, see src/core/precision.rs), then read the pack
# back with scripts/goz1_trit_histogram.py for exact trit counts.
#
# Weights and packs stay under ~/.models (never committed). The 3 GiB npy
# stage is a fresh mktemp dir on disk (not /tmp tmpfs) and is removed on exit.
#
# Usage:
#   scripts/tau_sweep_embedding.sh                 # default τ set
#   TAUS="0.05 0.2" scripts/tau_sweep_embedding.sh # custom τ set

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:-$HOME/.models/xai-grok-1/ckpt-0}"
ART="${ART:-$HOME/.models/xai-grok-1/artifacts}/tau-sweep"
TAUS="${TAUS:-0.0 0.05 0.1 0.2 0.3 0.5 0.7 1.0}"
BIN="$REPO/target/release/grok-ozempic"

if command -v gtime >/dev/null 2>&1; then
  TIME_V=(gtime -v)
elif [ "$(uname -s)" = Darwin ]; then
  TIME_V=(/usr/bin/time -l)
else
  TIME_V=(/usr/bin/time -v)
fi

[ -x "$BIN" ] || {
  echo "error: $BIN missing; run: cargo build --release --features cli --locked" >&2
  exit 1
}
[ -f "$CKPT/tensor00000_000" ] || {
  echo "error: embedding shard not found under $CKPT" >&2
  exit 1
}

# Fresh disk-backed stage; never wipe caller-owned dirs.
# Parent must exist before mktemp -d with a template path; ART/logs before tee.
mkdir -p "$HOME/.models/xai-grok-1" "$ART/logs"
WORK="$(mktemp -d "$HOME/.models/xai-grok-1/tau-sweep-stage.XXXXXX")"
STAGE="$WORK/npy"
mkdir -p "$STAGE"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "== export embedding npy -> $STAGE"
"${TIME_V[@]}" python3 "$REPO/scripts/export_grok1_embedding_npy.py" \
  --shard "$CKPT/tensor00000_000" \
  --output-dir "$STAGE" 2>&1 | tee "$ART/logs/export.log"

# Guard: exactly the sole embedding npy (quantize-goz1 packs every .npy).
NPYS=()
while IFS= read -r f; do
  [ -n "$f" ] && NPYS+=("$f")
done <<EOF
$(find "$STAGE" -maxdepth 1 -name '*.npy' | sort)
EOF
if [ "${#NPYS[@]}" -ne 1 ] ||
  [ "$(basename "${NPYS[0]}")" != "embedding__slot_00__token_embedding.npy" ]; then
  echo "error: expected sole embedding__slot_00__token_embedding.npy in $STAGE; found: ${NPYS[*]:-none}" >&2
  exit 1
fi

# Sweep manifest: strip every gif_threshold so CLI --gif-threshold wins.
# resolve_threshold precedence is per-tensor candidate > defaults > config;
# oz.gif_threshold metadata only records defaults||config, so per-tensor
# overrides would silently beat CLI and still pass the fidelity check.
MANI="$WORK/sweep-manifest.json"
python3 - "$REPO/dissect/grok-1/baseline.json" "$MANI" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    m = json.load(f)
m.setdefault("defaults", {}).pop("gif_threshold", None)
stripped = 0
for entry in m.get("ternary_candidates") or []:
    if isinstance(entry, dict) and "gif_threshold" in entry:
        entry.pop("gif_threshold", None)
        stripped += 1
with open(dst, "w") as f:
    json.dump(m, f, indent=2)
print(
    f"sweep manifest -> {dst} "
    f"(defaults: {m.get('defaults')}; stripped {stripped} per-tensor gif_threshold)"
)
PYEOF

for TAU in $TAUS; do
  OUT="$ART/goz1-tau-${TAU}.goz1"
  LOG="$ART/logs/tau-${TAU}.log"
  echo "== tau=$TAU -> $OUT"
  "${TIME_V[@]}" "$BIN" quantize-goz1 \
    --input-dir "$STAGE" \
    --output "$OUT" \
    --manifest "$MANI" \
    --input-format npy \
    --gif-threshold "$TAU" \
    --verify 2>&1 | tee "$LOG"
  # One pack analysis: JSON artifact + human summary (no double full-file scan).
  HIST_JSON="$ART/logs/tau-${TAU}.hist.json"
  python3 "$REPO/scripts/goz1_trit_histogram.py" "$OUT" \
    --json-out "$HIST_JSON" | tee -a "$LOG"

  # Enforce τ fidelity: pack metadata must match CLI TAU (f32); no invalid trits.
  python3 - "$HIST_JSON" "$TAU" <<'PYEOF' | tee -a "$LOG"
import json, struct, sys
path, tau_s = sys.argv[1], sys.argv[2]
# CLI parses --gif-threshold as f32; metadata serializes that f32.
want = struct.unpack("<f", struct.pack("<f", float(tau_s)))[0]
with open(path) as f:
    hist = json.load(f)
raw = hist.get("metadata", {}).get("oz.gif_threshold")
if raw is None:
    print(f"error: pack missing oz.gif_threshold metadata (wanted gif_threshold={want})", file=sys.stderr)
    sys.exit(1)
got = struct.unpack("<f", struct.pack("<f", float(raw)))[0]
if got != want:
    print(
        f"error: effective oz.gif_threshold={got} != requested TAU={want} "
        f"(manifest default/per-tensor may have overridden CLI)",
        file=sys.stderr,
    )
    sys.exit(1)
for t in hist.get("tensors", []):
    if t.get("type") != "ternary":
        continue
    inv = int(t.get("histogram", {}).get("invalid", 0))
    if inv:
        print(f"error: tensor {t['name']} has {inv} invalid trit codes", file=sys.stderr)
        sys.exit(1)
    zeros_pct = 100.0 * t["histogram"]["zeros"] / t["num_elements"] if t["num_elements"] else 0.0
    print(f"  ok: oz.gif_threshold={got} matches TAU={want}; zeros%={zeros_pct:.4f}")
PYEOF
done

echo "== done; packs + logs under $ART"
