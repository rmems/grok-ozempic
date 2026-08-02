# xai-dissect cartography handoff

Checked-in snapshot of the latest **correct** Grok-1 structural campaign from
[`xai-dissect`](https://github.com/rmems/xai-dissect). This repo does not produce
these artifacts; it consumes them for planning and SAAQ handoff.

## Current pointer

| File | Role |
|------|------|
| `LATEST_CORRECT_GROK1_RUN` | Symlink → active run directory |
| `LATEST_CORRECT_GROK1_RUN.txt` | Human-readable summary (Windows-safe) |
| `grok1_run3_20260802T023050Z/` | Full run3 tree (exports, manifests, reports, log) |

Slug: `xai-grok-1-ckpt-0` · Checkpoint: `~/.models/xai-grok-1/ckpt-0` · 770 tensors.

## Prefer these over May run2

Do **not** use retired `reports/grok-1-official__ckpt-0/` paths. Primary entry
points:

- `LATEST_CORRECT_GROK1_RUN/reports/xai-grok-1-ckpt-0/quant-plan.md`
- `LATEST_CORRECT_GROK1_RUN/manifests/xai-grok-1-ckpt-0/conversion-manifest.json`
- `LATEST_CORRECT_GROK1_RUN/manifests/xai-grok-1-ckpt-0/pilot-selection-plan.json`
- Comparison: [`docs/runs/grok1_run3_vs_run2_comparison.md`](../docs/runs/grok1_run3_vs_run2_comparison.md)

## Refresh

From a machine with the xai-dissect branch checked out:

```bash
SRC=~/rmems/xai-dissect   # branch run/grok1-run3-20260802 (or later LATEST)
rsync -a --delete "$SRC/out/grok1_run3_20260802T023050Z/" out/grok1_run3_20260802T023050Z/
cp -f "$SRC/out/LATEST_CORRECT_GROK1_RUN.txt" out/
ln -sfn grok1_run3_20260802T023050Z out/LATEST_CORRECT_GROK1_RUN
cp -f "$SRC/docs/runs/grok1_run3_vs_run2_comparison.md" docs/runs/
```

## Not for `quantize-goz1 --manifest`

`conversion-manifest.json` / `quant-plan.json` are **not** the
`xai-dissect.manifest` schema used by `stream::resolve_manifest`. Runtime packs
still use `dissect/grok-1/baseline.json` (V1) until #40 / RM-191; structural V2
fixture remains `dissect/grok-1/structural-manifest.json` for alignment/dry-run.
