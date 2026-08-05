#!/usr/bin/env bash
# Bounded single-block Grok-1 GOZ1 pilot — GH #53 / RM-222, beads goz-4ic2.
#
# Consumes the xai-dissect run3 planning surface (conversion-manifest +
# quant-plan + pilot-selection-plan), dequantizes the block's int8 attention /
# MoE tensors to f32 npy, packs one GOZ1 under a **tier-aware** V2 structural
# manifest, verifies it, and measures route-preservation metrics.
#
# Policy enforced here (run3 quant-plan `keep_fp32` + #51 per-tier τ):
#   preserve : router, block_norm, final_norm   — never ternary, no τ
#   ternary  : attn_proj_i8.* @ τ=$TAU_ATTN, moe_expert.* @ τ=$TAU_EXPERT
#   deferred : token_embedding (explicitly NOT a candidate here, so a stray
#              embedding npy hard-errors instead of being packed)
#
# The pilot manifest is derived at runtime from the in-tree V2 structural
# manifest; nothing under dissect/ is modified (xai-dissect stays authoritative).
# defaults.gif_threshold is stripped and per-tensor τ set instead, because
# defaults silently override CLI --gif-threshold (the #51 τ trap,
# src/core/precision.rs::resolve_threshold).
#
# Weights, npy stages and packs stay under ~/.models — never committed.
#
# Usage:
#   scripts/block_pilot_goz1.sh                          # block 0, attention_only
#   BLOCK=0 MODE=attention_plus_expert scripts/block_pilot_goz1.sh
#   KEEP_NPY=1 scripts/block_pilot_goz1.sh               # keep the f32 stage

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCK="${BLOCK:-0}"
MODE="${MODE:-attention_only}"
TAU_ATTN="${TAU_ATTN:-0.4}"
TAU_EXPERT="${TAU_EXPERT:-0.9}"
DISSECT_RUN="${GROK_OZEMPIC_DISSECT_RUN:-$HOME/rmems/xai-dissect/out/LATEST_CORRECT_GROK1_RUN}"
RUN3="$DISSECT_RUN/manifests/xai-grok-1-ckpt-0"
ART="${ART:-$HOME/.models/xai-grok-1/artifacts}/block-pilot"
BIN="$REPO/target/release/grok-ozempic"
BLOCK_LABEL="$(printf 'block_%03d' "$BLOCK")"
TAG="${BLOCK_LABEL}-${MODE}"

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
[ -f "$RUN3/conversion-manifest.json" ] || {
  echo "error: run3 conversion-manifest not found under $RUN3" >&2
  echo "       set GROK_OZEMPIC_DISSECT_RUN to the xai-dissect run root" >&2
  exit 1
}

mkdir -p "$HOME/.models/xai-grok-1" "$ART/logs"
WORK="$(mktemp -d "$HOME/.models/xai-grok-1/block-pilot-stage.XXXXXX")"
STAGE="$WORK/npy"
mkdir -p "$STAGE"
cleanup() {
  if [ -n "${KEEP_NPY:-}" ]; then
    echo "== npy stage kept at $STAGE"
  else
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT

echo "== [1/5] dequant export: $BLOCK_LABEL / $MODE -> $STAGE"
"${TIME_V[@]}" python3 "$REPO/scripts/export_grok1_int8_npy.py" \
  --conversion-manifest "$RUN3/conversion-manifest.json" \
  --block "$BLOCK" --mode "$MODE" \
  --output-dir "$STAGE" 2>&1 | tee "$ART/logs/$TAG-export.log"

echo "== [2/5] derive tier-aware V2 pilot manifest"
MANIFEST="$ART/$TAG-manifest.json"
python3 - "$REPO/dissect/grok-1/structural-manifest.json" "$MANIFEST" \
  "$TAU_ATTN" "$TAU_EXPERT" "$RUN3/quant-plan.json" <<'PY'
import json, sys

src, dst, tau_attn, tau_expert, quant_plan_path = sys.argv[1:6]
m = json.load(open(src))
plan = json.load(open(quant_plan_path))

# run3 quant-plan is the authority for which families may be touched.
keep = set(plan["keep_fp32"])
pilot = set(plan["pilot_quantize"])
deferred = set(plan["defer"])

def kind_of(pattern):
    # "block_*.slot_04.attn_proj_i8.model_width" -> "attn_proj_i8.model_width"
    return pattern.split(".", 2)[2] if pattern.count(".") >= 2 else pattern

# Preserve list stays exactly as authored upstream, but assert it only covers
# families run3 marked keep_fp32 — a drifted preserve rule must not slip through.
for e in m["preserve"]:
    k = kind_of(e["name"])
    if k not in keep:
        sys.exit(f"error: preserve rule {e['name']!r} (kind {k!r}) not in quant-plan keep_fp32 {sorted(keep)}")

cands = []
for e in m["ternary_candidates"]:
    k = kind_of(e["name"])
    if k in deferred:
        continue  # token_embedding: deferred, and dropped so a stray npy hard-errors
    if k not in pilot:
        sys.exit(f"error: ternary candidate {e['name']!r} (kind {k!r}) not in quant-plan pilot_quantize")
    tau = float(tau_expert) if k.startswith("moe_expert") else float(tau_attn)
    cands.append({**e, "gif_threshold": tau})

m["ternary_candidates"] = cands
# Strip the τ trap: defaults.gif_threshold outranks CLI --gif-threshold.
m["defaults"].pop("gif_threshold", None)
m["produced_by"] = {
    "tool": "grok-ozempic scripts/block_pilot_goz1.sh (derived; xai-dissect remains authoritative)",
    "version": m["produced_by"].get("version"),
    "commit": None,
}
json.dump(m, open(dst, "w"), indent=2)
print(f"derived {dst}: {len(m['preserve'])} preserve, {len(cands)} ternary candidate patterns")
for c in cands:
    print(f"  ternary {c['name']:<44} tau={c['gif_threshold']}")
PY

echo "== [3/5] pack GOZ1 (--verify)"
PACK="$ART/$TAG.goz1"
"${TIME_V[@]}" "$BIN" quantize-goz1 \
  --input-dir "$STAGE" \
  --output "$PACK" \
  --manifest "$MANIFEST" \
  --input-format npy \
  --verify 2>&1 | tee "$ART/logs/$TAG-pack.log"

echo "== [4/5] exact trit histogram"
python3 "$REPO/scripts/goz1_trit_histogram.py" "$PACK" \
  --json-out "$ART/$TAG-histogram.json" 2>&1 | tee "$ART/logs/$TAG-histogram.log"

echo "== [5/5] route-preservation metrics"
python3 "$REPO/scripts/route_preservation_metrics.py" \
  --npy-dir "$STAGE" \
  --pack "$PACK" \
  --block "$BLOCK" \
  --mode "$MODE" \
  --json-out "$ART/$TAG-route-preservation.json" 2>&1 |
  tee "$ART/logs/$TAG-metrics.log"

echo
echo "== pilot artifacts"
echo "   manifest   $MANIFEST"
echo "   pack       $PACK"
echo "   histogram  $ART/$TAG-histogram.json"
echo "   metrics    $ART/$TAG-route-preservation.json"
echo "   logs       $ART/logs/$TAG-*.log"
