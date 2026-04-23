# dissect/ — non-authoritative reference manifests

This directory contains **non-authoritative** reference manifests consumed
by `grok-ozempic`'s manifest loader (see
[`src/core/manifest.rs`](../src/core/manifest.rs) and
[`docs/dissect-manifest.md`](../docs/dissect-manifest.md)).

## Source of truth

The authoritative source for these manifests is the upstream
[`xai-dissect`](https://github.com/rmems/xai-dissect) repository. The
files checked in here are **regenerated fallback / reference copies**
intended for:

- bootstrapping (so the crate can be built and tested without first
  running `xai-dissect`),
- documentation (a minimal example of schema v1 in practice),
- fallback resolution when no explicit manifest path is supplied.

Do **not** hand-edit these files to drive pipeline behavior in
production runs. If `xai-dissect` says something different, `xai-dissect`
wins. This repo does not produce manifests; it only consumes them.

## Layout

- `grok-1/baseline.json` — minimal v1 manifest for Grok-1 (xai-org/grok-1)
  encoding today's default router/gate heuristic as a `preserve` list.

## Consumption order (phase 2+)

Resolved first-hit-wins by the selection seam introduced in phase 2:

1. `QuantizationConfig.manifest_path` (explicit caller override).
2. `GROK_OZEMPIC_MANIFEST` environment variable.
3. In-tree fallback (this directory, `grok-1/baseline.json`).
4. Legacy `router_patterns` substring heuristic in `stream.rs`.

Phase 1 (current) only loads manifests on demand; the pipeline is not
wired yet.
