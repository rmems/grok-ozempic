#!/usr/bin/env bash
# Bounded single-block Grok-1 GOZ1 pilot — GH #53 / RM-222, beads goz-4ic2.
#
# Consumes the xai-dissect run3 planning surface (conversion-manifest +
# quant-plan + pilot-selection-plan), dequantizes the block's int8 attention /
# MoE tensors to f32 npy, packs one GOZ1 under a **tier-aware** V2 structural
# manifest, verifies it, validates the pack counters against the selected mode,
# and measures route-preservation metrics.
#
# Policy enforced here (run3 quant-plan `keep_fp32` + #51 per-tier τ):
#   preserve : router, block_norm, final_norm   — never ternary, no τ
#   ternary  : attn_proj_i8.* @ τ=$TAU_ATTN, moe_expert.* @ τ=$TAU_EXPERT
#   deferred : token_embedding (explicitly NOT a candidate here, so a stray
#              embedding npy hard-errors instead of being packed)
#
# The pilot manifest is derived at runtime from the in-tree V2 structural
# manifest; nothing under dissect/ is modified (xai-dissect stays authoritative).
# The deriver is fail-closed: unknown ternary families abort rather than silently
# inheriting the attention τ. defaults.gif_threshold is stripped and per-tensor
# τ set instead, because
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
DISSECT_RUN="${GROK_OZEMPIC_DISSECT_RUN:-$HOME/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN}"
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
for required in conversion-manifest.json quant-plan.json pilot-selection-plan.json; do
  [ -f "$RUN3/$required" ] || {
    echo "error: run3 $required not found under $RUN3" >&2
    echo "       set GROK_OZEMPIC_DISSECT_RUN to the xai-dissect run root" >&2
    exit 1
  }
done

echo "== [0/5] validate BLOCK/MODE against run3 pilot-selection-plan"
python3 - "$RUN3/pilot-selection-plan.json" "$BLOCK" "$MODE" <<'PY'
import json, sys
path, block, mode = sys.argv[1:4]
plan = json.load(open(path))
allowed = []
if isinstance(plan, list):
    allowed = [(p.get("block"), p.get("mode")) for p in plan if isinstance(p, dict)]
elif isinstance(plan, dict):
    # run3 schema: selected_blocks[{block_index, ...}] x modes[...]
    selected_blocks = plan.get("selected_blocks")
    modes = plan.get("modes")
    if isinstance(selected_blocks, list) and isinstance(modes, list):
        allowed = [
            (p.get("block_index"), m)
            for p in selected_blocks
            if isinstance(p, dict)
            for m in modes
        ]
    else:
        for key in ("pilots", "selected", "runs", "matrix"):
            if key in plan and isinstance(plan[key], list):
                allowed = [(p.get("block"), p.get("mode")) for p in plan[key] if isinstance(p, dict)]
                break
    # No last-ditch "scrape every top-level key" fallback: it could synthesize a
    # (block, mode) pair the plan never authorized (any numeric-keyed list of
    # strings would validate), which is exactly what this fail-closed gate is
    # here to prevent. An unrecognized schema must abort, not guess.
if not allowed:
    sys.exit(
        f"error: {path} has no recognizable pilot matrix "
        "(expected `selected_blocks` + `modes`, or a list of {block, mode} objects); "
        "refusing to validate BLOCK/MODE against an unknown schema"
    )
try:
    block_val = int(block)
except ValueError:
    block_val = block
if (block_val, mode) not in allowed:
    sys.exit(f"error: BLOCK={block} MODE={mode} not in pilot-selection-plan ({path}); allowed: {sorted(set(allowed))}")
PY

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
set +e
"${TIME_V[@]}" python3 "$REPO/scripts/export_grok1_int8_npy.py" \
  --conversion-manifest "$RUN3/conversion-manifest.json" \
  --block "$BLOCK" --mode "$MODE" \
  --output-dir "$STAGE" 2>&1 | tee "$ART/logs/$TAG-export.log"
PIPESTATUS_COPY=("${PIPESTATUS[@]}")
set -e
EXPORT_RC=${PIPESTATUS_COPY[0]}
EXPORT_TEE_RC=${PIPESTATUS_COPY[1]}
if [ "$EXPORT_TEE_RC" -ne 0 ]; then
  echo "error: export log tee failed (exit $EXPORT_TEE_RC)" >&2
  exit "$EXPORT_TEE_RC"
fi
if [ "$EXPORT_RC" -ne 0 ]; then
  echo "error: export failed (exit $EXPORT_RC)" >&2
  exit "$EXPORT_RC"
fi

echo "== [2/5] derive tier-aware V2 pilot manifest"
MANIFEST="$ART/$TAG-manifest.json"
python3 - "$REPO/dissect/grok-1/structural-manifest.json" "$MANIFEST" \
  "$TAU_ATTN" "$TAU_EXPERT" "$RUN3/quant-plan.json" <<'PY'
import json, sys

src, dst, tau_attn, tau_expert, quant_plan_path = sys.argv[1:6]
m = json.load(open(src))
plan = json.load(open(quant_plan_path))

def require(doc, key, where):
    if key not in doc:
        sys.exit(f"error: {where} has no {key!r} key (schema drift?); present: {sorted(doc)}")
    return doc[key]

# run3 quant-plan is the authority for which families may be touched.
keep = set(require(plan, "keep_fp32", quant_plan_path))
pilot = set(require(plan, "pilot_quantize", quant_plan_path))
deferred = set(require(plan, "defer", quant_plan_path))
for key in ("preserve", "ternary_candidates", "defaults", "produced_by"):
    require(m, key, src)

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
    if k.startswith("moe_expert"):
        tau = float(tau_expert)
    elif k.startswith("attn_proj_i8"):
        tau = float(tau_attn)
    else:
        sys.exit(f"error: ternary candidate {e['name']!r} (kind {k!r}) has no defined τ; supported families are attn_proj_i8.* and moe_expert.*")
    cands.append({**e, "gif_threshold": tau})

m["ternary_candidates"] = cands
# Strip the τ trap: defaults.gif_threshold outranks CLI --gif-threshold.
m["defaults"].pop("gif_threshold", None)
# The Rust side declares produced_by.version as String with serde(default): an
# absent key is fine, an explicit null is not. So omit it when upstream omits it,
# and only reject a present-but-non-string value.
produced_by = {
    "tool": "grok-ozempic scripts/block_pilot_goz1.sh (derived; xai-dissect remains authoritative)",
    "commit": None,
}
if "version" in m["produced_by"]:
    version = m["produced_by"]["version"]
    if not isinstance(version, str):
        sys.exit(
            f"error: {src} produced_by.version is {version!r}, expected a string; "
            "emitting null here would make the derived manifest fail deserialization"
        )
    produced_by["version"] = version
m["produced_by"] = produced_by
with open(dst, "w") as fh:
    json.dump(m, fh, indent=2)
    fh.write("\n")
print(f"derived {dst}: {len(m['preserve'])} preserve, {len(cands)} ternary candidate patterns")
for c in cands:
    print(f"  ternary {c['name']:<44} tau={c['gif_threshold']}")
PY

echo "== [3/5] pack GOZ1 (--verify)"
PACK="$ART/$TAG.goz1"
set +e
"${TIME_V[@]}" "$BIN" quantize-goz1 \
  --input-dir "$STAGE" \
  --output "$PACK" \
  --manifest "$MANIFEST" \
  --input-format npy \
  --verify 2>&1 | tee "$ART/logs/$TAG-pack.log"
PIPESTATUS_COPY=("${PIPESTATUS[@]}")
set -e
PACK_RC=${PIPESTATUS_COPY[0]}
PACK_TEE_RC=${PIPESTATUS_COPY[1]}
if [ "$PACK_TEE_RC" -ne 0 ]; then
  echo "error: pack log tee failed (exit $PACK_TEE_RC)" >&2
  exit "$PACK_TEE_RC"
fi
if [ "$PACK_RC" -ne 0 ]; then
  echo "error: GOZ1 pack failed (exit $PACK_RC)" >&2
  exit "$PACK_RC"
fi

echo "== [4/5] exact trit histogram"
set +e
python3 "$REPO/scripts/goz1_trit_histogram.py" "$PACK" \
  --json-out "$ART/$TAG-histogram.json" 2>&1 | tee "$ART/logs/$TAG-histogram.log"
PIPESTATUS_COPY=("${PIPESTATUS[@]}")
set -e
HIST_RC=${PIPESTATUS_COPY[0]}
HIST_TEE_RC=${PIPESTATUS_COPY[1]}
if [ "$HIST_TEE_RC" -ne 0 ]; then
  echo "error: histogram log tee failed (exit $HIST_TEE_RC)" >&2
  exit "$HIST_TEE_RC"
fi
if [ "$HIST_RC" -ne 0 ]; then
  echo "error: histogram failed (exit $HIST_RC)" >&2
  exit "$HIST_RC"
fi

echo "== [4a/5] annotate histogram with per-tensor effective τ"
python3 - "$ART/$TAG-histogram.json" "$TAU_ATTN" "$TAU_EXPERT" <<'PY'
import json, sys
hist_path, tau_attn, tau_expert = sys.argv[1:4]
hist = json.load(open(hist_path))
def kind_of(name):
    return name.split(".", 2)[2] if name.count(".") >= 2 else name
# Never invent a value: if the pack recorded no oz.gif_threshold, leave the key
# absent so a reader can tell "pack said nothing" from "pack said X".
default = hist["metadata"].get("oz.gif_threshold")
pipeline = f"pipeline default {default}" if default is not None else "no pipeline default recorded"
hist["metadata"]["oz.gif_threshold_note"] = (
    f"per-tensor effective_tau: {tau_attn} for attn_proj_i8.*, "
    f"{tau_expert} for moe_expert.*; {pipeline}"
)
for t in hist["tensors"]:
    if t["type"] != "ternary":
        continue
    k = kind_of(t["name"])
    if k.startswith("attn_proj_i8"):
        t["effective_tau"] = float(tau_attn)
    elif k.startswith("moe_expert"):
        t["effective_tau"] = float(tau_expert)
json.dump(hist, open(hist_path, "w"), indent=2)
print(f"annotated {hist_path} with per-tensor effective_tau")
PY

echo "== [4b/5] validate pack counters against selected mode"
HIST="$ART/$TAG-histogram.json"
python3 - "$RUN3/conversion-manifest.json" "$HIST" "$BLOCK" "$MODE" "$REPO/scripts" <<'PY'
import json, sys
from pathlib import Path
conv_path, hist_path, block, mode, scripts_dir = sys.argv[1:6]
block = int(block)
sys.path.insert(0, scripts_dir)
from export_grok1_int8_npy import MODES
PRESERVE = {"router", "block_norm", "final_norm"}
conv = json.load(open(conv_path))
hist = json.load(open(hist_path))
ternary_kinds = set(MODES[mode]) - PRESERVE
expected_ternary = sum(1 for t in conv["tensors"] if t.get("block") == block and t.get("kind") in ternary_kinds)
expected_preserve = sum(1 for t in conv["tensors"] if t.get("block") == block and t.get("kind") in PRESERVE)
actual_ternary = sum(1 for t in hist["tensors"] if t["type"] == "ternary")
actual_preserve = sum(1 for t in hist["tensors"] if t["type"] == "f16")
if (actual_ternary, actual_preserve) != (expected_ternary, expected_preserve):
    sys.exit(f"error: pack counter mismatch for {mode} block {block}: expected {expected_ternary} ternary + {expected_preserve} preserve, got {actual_ternary} + {actual_preserve}")
print(f"pack counters ok: {actual_ternary} ternary, {actual_preserve} preserve")
PY

echo "== [5/5] route-preservation metrics"
# Exit 3 means a gate did not pass. That is a legitimate pilot outcome (the whole
# point is to measure), so it is reported rather than aborting the run; any other
# non-zero code is a real failure and still stops the script.
# `|| true` would reset PIPESTATUS, so disable errexit around the pipeline instead.
set +e
python3 "$REPO/scripts/route_preservation_metrics.py" \
  --npy-dir "$STAGE" \
  --pack "$PACK" \
  --conversion-manifest "$RUN3/conversion-manifest.json" \
  --block "$BLOCK" \
  --mode "$MODE" \
  --json-out "$ART/$TAG-route-preservation.json" 2>&1 |
  tee "$ART/logs/$TAG-metrics.log"
PIPESTATUS_COPY=("${PIPESTATUS[@]}")
set -e
METRICS_RC=${PIPESTATUS_COPY[0]}
TEE_RC=${PIPESTATUS_COPY[1]}
if [ "$TEE_RC" -ne 0 ]; then
  echo "error: metrics log tee failed (exit $TEE_RC)" >&2
  exit "$TEE_RC"
fi
if [ "$METRICS_RC" -eq 3 ]; then
  echo "note: route-preservation gates did not all pass (exit 3) — recorded, not fatal"
elif [ "$METRICS_RC" -ne 0 ]; then
  echo "error: route-preservation metrics failed (exit $METRICS_RC)" >&2
  exit "$METRICS_RC"
fi

echo "== [5b/5] annotate route-preservation JSON with per-tensor effective τ"
python3 - "$ART/$TAG-route-preservation.json" "$TAU_ATTN" "$TAU_EXPERT" <<'PY'
import json, sys
path, tau_attn, tau_expert = sys.argv[1:4]
rep = json.load(open(path))
def kind_of(name):
    return name.split(".", 2)[2] if name.count(".") >= 2 else name
md = rep.setdefault("pilot", {}).setdefault("pack_metadata", {})
# Never invent a value: if the pack recorded no oz.gif_threshold, leave the key
# absent so a reader can tell "pack said nothing" from "pack said X".
default = md.get("oz.gif_threshold")
pipeline = f"pipeline default {default}" if default is not None else "no pipeline default recorded"
md["oz.gif_threshold_note"] = (
    f"per-tensor effective_tau: {tau_attn} for attn_proj_i8.*, "
    f"{tau_expert} for moe_expert.*; {pipeline}"
)
rep["pilot"]["effective_tau_policy"] = {
    "attn_proj_i8.*": float(tau_attn),
    "moe_expert.*": float(tau_expert),
}
for section in ("weights", "routing"):
    for name, entry in rep.get(section, {}).items():
        k = kind_of(name)
        if k.startswith("attn_proj_i8"):
            entry["effective_tau"] = float(tau_attn)
        elif k.startswith("moe_expert"):
            entry["effective_tau"] = float(tau_expert)
json.dump(rep, open(path, "w"), indent=2)
print(f"annotated {path} with per-tensor effective_tau")
PY

echo "== [5c/5] write publish copies with host-local paths reduced"
python3 - "$ART/$TAG-histogram.json" "$ART/$TAG-histogram.publish.json" <<'PY'
import json, os, sys

src, dst = sys.argv[1:3]
hist = json.load(open(src))
# `file` is an absolute path under ~/.models that cannot resolve elsewhere.
if "file" in hist:
    hist["file_basename"] = os.path.basename(hist.pop("file"))
with open(dst, "w") as fh:
    json.dump(hist, fh, indent=2)
    fh.write("\n")
print(f"wrote publish copy {dst}")
PY

python3 - "$ART/$TAG-route-preservation.json" "$ART/$TAG-route-preservation.publish.json" <<'PY'
import json, os, sys

src, dst = sys.argv[1:3]
rep = json.load(open(src))
pilot = rep.get("pilot", {})
# The working copy keeps absolute paths (useful while debugging on this host).
# The publish copy is what belongs under reports/: the npy stage is a mktemp dir
# that no longer exists after the run, and an absolute /home/<user>/... path
# cannot resolve on any other machine, so committing it implies a
# reproducibility that does not exist. Basenames + the documented driver command
# are the honest provenance.
for key in ("pack", "npy_dir"):
    if key in pilot:
        pilot[f"{key}_basename"] = os.path.basename(str(pilot.pop(key)).rstrip("/"))
pilot["paths_note"] = (
    "Host-local absolute paths removed for publication; regenerate with "
    "BLOCK=<n> MODE=<mode> scripts/block_pilot_goz1.sh (the npy stage is a "
    "per-run mktemp directory deleted on exit)."
)
with open(dst, "w") as fh:
    json.dump(rep, fh, indent=2)
    fh.write("\n")
print(f"wrote publish copy {dst}")
PY

echo
echo "== pilot artifacts"
echo "   manifest   $MANIFEST"
echo "   pack       $PACK"
echo "   histogram  $ART/$TAG-histogram.json"
echo "   publish    $ART/$TAG-histogram.publish.json  (copy this under reports/)"
echo "   metrics    $ART/$TAG-route-preservation.json"
echo "   publish    $ART/$TAG-route-preservation.publish.json  (copy this under reports/)"
echo "   logs       $ART/logs/$TAG-*.log"
