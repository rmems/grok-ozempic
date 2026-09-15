# Architecture: kernel ownership boundary

This document is the ownership map between `grok-ozempic` and
[`myelin-accelerator`](https://github.com/Limen-Neural/myelin-accelerator).
The boundary landed in
[PR #25](https://github.com/rmems/grok-ozempic/pull/25)
([#21](https://github.com/rmems/grok-ozempic/issues/21)).
Docs that walk a new reader through it are tracked as
[#20](https://github.com/rmems/grok-ozempic/issues/20) /
[RM-64](https://linear.app/rpd-34/issue/RM-64).

`grok-ozempic` is the **Grok-1-specific quantization and orchestration layer**.
It must not grow a duplicated CUDA kernel stack unless a kernel is truly
Grok-specific. Kernel ownership lives in `myelin-accelerator` so that
binary/ternary/SAAQ kernels, bitpacking, benchmarks, and FFI stay reusable
across `corinth-canal`, Grok-1 experiments, and future Metis/Spikenaut work.

---

## Five-minute path: manifest → selection → backend → myelin

Start here. The rest of this file is the ownership table and the current vs
future dispatch caveat.

```
xai-dissect manifest (JSON)
        │  preserve > fp16 > ternary_candidates > defaults
        ▼
stream::resolve_manifest  +  selection.rs / precision::decide
        │  V2 unmatched names hard-error (no defaults fallthrough)
        ▼
DryRunPlanner  (src/core/dry_run.rs)
        │  OperationKind per rule + CoverageSummary vs 770-tensor inventory
        ▼
BackendKernel trait  (src/core/backend.rs)
        ├── LocalBackend     → quantizer.rs (CPU, current math)
        └── MyelinBackend    → myelin-accelerator FFI (stub until linked)
                ▼
        Limen-Neural/myelin-accelerator
            host bitpacking public; packed ternary GEMV/GEMM on the kernel side
```

| Step | What to open | What it does |
|------|----------------|--------------|
| 1. Manifest | [`dissect-manifest.md`](./dissect-manifest.md), `dissect/grok-1/structural-manifest.json` | Assigns `preserve` / `fp16` / `ternary_snn`. `xai-dissect` is authoritative; the in-tree copies are policy references. |
| 2. Selection | `src/core/selection.rs`, `src/core/precision.rs` | First matching rule wins. V2 requires structural-named inputs and fails closed on a miss. |
| 3. Dry-run plan | [`src/core/dry_run.rs`](../src/core/dry_run.rs) | Maps each rule to an [`OperationKind`](#operationkind) and reports inventory coverage. **No weight payloads.** Distinct from CLI `--dry-run` on `smoke-grok1` / `convert-grok1` (those write SAAQ metadata indexes). |
| 4. Backend trait | [`src/core/backend.rs`](../src/core/backend.rs) | `quantize_f32`, `quantize_f16`, `pack_ternary`, `passthrough_f16`, `convert_f32_to_f16_bytes`. |
| 5. Kernels | [Limen-Neural/myelin-accelerator](https://github.com/Limen-Neural/myelin-accelerator) | Reusable CUDA / packed GEMV. Route new kernel work there, not here. |

**Today's `quantize-goz1` path does not dispatch through the trait.**
`run_quantization` in [`src/core/stream.rs`](../src/core/stream.rs) still calls
`quantizer::quantize_f32` / FP16 helpers directly. The math is the same as
`LocalBackend`; the FFI seam is not on the hot path. Do not report myelin or
CUDA as the backend for a GOZ1 pack. See
[`first-quantization-target.md`](./first-quantization-target.md) and the
[README backend section](../README.md#backend-and-kernel-boundary).

---

## Ownership table

| Area | Owner | Reason |
|------|-------|--------|
| Grok-1 checkpoint / shard handling | `grok-ozempic` | Grok-1 shard naming, safetensors / NPY layout |
| Tensor inventory and mapping | `grok-ozempic` | Manifest-driven precision classification |
| Router/expert-aware quantization planning | `grok-ozempic` | Grok-1 MoE structure |
| Per-expert quantization manifests | `grok-ozempic` | xai-dissect integration |
| Validation against xai-dissect artifacts | `grok-ozempic` | Grok-1 artifact contract |
| Dry-run quantization reports | `grok-ozempic` | Orchestration concern (`DryRunPlanner`) |
| High-level experiment orchestration | `grok-ozempic` | Pipeline entry points |
| GOZ1 binary container format | `grok-ozempic` | Grok-specific output format ([`goz1-format.md`](./goz1-format.md)) |
| CPU `pack_trits` / `encode_trit` | `grok-ozempic` (`quantizer.rs`) | Current `LocalBackend` / stream path; same math, not CUDA |
| Host / CUDA bitpacking | **`myelin-accelerator`** | Reusable `bitpacking`; do not grow a second CUDA packer here |
| Packed ternary GEMV / GEMM | `myelin-accelerator` | Generic CUDA kernel (myelin [#9](https://github.com/Limen-Neural/myelin-accelerator/issues/9)) |
| SAAQ routing / reduction kernels | `myelin-accelerator` | Generic CUDA kernel |
| Kernel benchmarks | `myelin-accelerator` | Reusable infrastructure (`just bench` here exits until a harness exists) |
| Rust/CUDA FFI and launch helpers | `myelin-accelerator` | Reusable infrastructure |

---

## Backend integration layer

The `BackendKernel` trait in [`src/core/backend.rs`](../src/core/backend.rs)
is the interface `grok-ozempic` will use for deployable kernel operations.
Both implementations are re-exported from the crate root.

```
grok-ozempic (orchestration)
    │
    ├── BackendKernel trait (src/core/backend.rs)
    │       │
    │       ├── LocalBackend       — delegates to quantizer.rs (CPU, current)
    │       └── MyelinBackend      — FFI to myelin-accelerator (stub; every
    │                                 method returns BackendNotAvailable)
    │
    ├── DryRunPlanner (src/core/dry_run.rs)
    │       — maps each manifest rule to an OperationKind / planned call
    │
    └── Live pack path today: stream.rs → quantizer.rs (not via the trait)
```

### `LocalBackend` (current math)

Wraps the existing Rust implementations in
[`src/core/quantizer.rs`](../src/core/quantizer.rs). Tests assert it matches
calling those functions directly. Use this type when writing new callers that
should be backend-swappable.

### `MyelinBackend` (stub)

Every method returns `GrokOzempicError::BackendNotAvailable`. The type exists
so callers can be written against the CUDA seam before the library is linked.
The intended dependency is
[`Limen-Neural/myelin-accelerator`](https://github.com/Limen-Neural/myelin-accelerator)
(not a second kernel tree in this repo). Host packing is already public there
as `bitpacking`. Device ternary matmul landed under myelin
[#9](https://github.com/Limen-Neural/myelin-accelerator/issues/9). Wiring a
feature-gated Cargo dep and replacing the stub is later work; until then
`quantize-goz1` stays on CPU.

---

## Dry-run planner

`DryRunPlanner::plan` reads an xai-dissect manifest plus a `ModelInventory`
and produces a `DryRunReport`:

1. **Validation** — planned calls vs the 770-tensor Grok-1 inventory
   (`CoverageStatus::Full` / `Partial` / `OverComplete`).
2. **Backend readiness** — which `OperationKind` counts a real `BackendKernel`
   must provide before the live pack path can switch off direct
   `quantizer.rs` calls.

`DryRunPlanner::planned_backend_calls_json` is the machine-readable form.
Each manifest matcher maps to `{operation, precision, gif_threshold,
estimated_tensor_count, class}`. The reserved key `__coverage__` holds
`by_operation`, `covered_by_rules`, `inventory_total`, `coverage`, and
`backend_handled_total`. See
[`artifact-compatibility-plan.md`](./artifact-compatibility-plan.md) for how
that output sits next to the inventory / routing reports.

Alignment tests in `src/core/alignment.rs` already run the structural
manifest through this planner. CLI `--dry-run` on the SAAQ metadata commands
is a different switch: it writes `saaq-g1-v0` indexes without packing weights.
Do not treat a green metadata report as a weight-fidelity experiment.

### `OperationKind`

This is an orchestration verb, not a 1:1 `BackendKernel` method list.
Wrapping an already-quantized source is an artifact-path operation, not a
kernel. Convert and quantize map onto real trait methods.

| `OperationKind` | When | BackendKernel method (when dispatched) |
|-----------------|------|------------------------------------------|
| `quantize_ternary` | `ternary_snn` on a float source (`f32` / `f16` / `bf16`) | `quantize_f32` (or `quantize_f16`) |
| `convert_fp16` | `preserve` or `fp16` | `convert_f32_to_f16_bytes` / `passthrough_f16` |
| `wrap_existing_quantized` | `ternary_snn` whose inventory dtype is already `i8` / `int8` / `u8` | none — wrap, do not re-quantize |

Wrap vs re-quantize is decided from inventory dtype, never from a glob
substring. Mixed dtypes on one matcher fail closed
(`GrokOzempicError::MixedInventoryDtype`).

---

## What stays local (adapter/glue code)

Some tensor layout transforms are Grok-1-specific enough that they warrant
local glue rather than backend kernel implementations:

| Transform | Justification |
|-----------|---------------|
| GOZ1 tensor table assembly | Grok-specific container format |
| Per-expert quantization manifest serialization | Grok-1 MoE structure |
| `ManifestEntry` → `PackTensorHeader` mapping | Combines manifest classification + GOZ1 format |
| Source dtype detection (F32/F16/BF16) | Input format handling, not kernel logic |

---

## Dependency status

`myelin-accelerator` is recorded as a **planned dependency** in
[`Cargo.toml`](../Cargo.toml) (commented out). It becomes a real, feature-gated
dependency once the `MyelinBackend` FFI bridge is implemented against
[Limen-Neural/myelin-accelerator](https://github.com/Limen-Neural/myelin-accelerator).

There is no 2026-05-28 sprint cutoff and no in-tree GPU provisioning runbook.
Kernel work and cloud GPU experiments belong in `myelin-accelerator` and
measured reports, not in this crate's default `just ci` path.

---

## Related docs

| Doc | Role |
|-----|------|
| [README](../README.md#backend-and-kernel-boundary) | Entry point: backend trait, measured status, CLI paths |
| [`dissect-manifest.md`](./dissect-manifest.md) | Manifest schema and precision → `OperationKind` mapping |
| [`grok1-saaq-artifact-flow.md`](./grok1-saaq-artifact-flow.md) | Copyable runbook, including backend call flow |
| [`first-quantization-target.md`](./first-quantization-target.md) | First real-weight pack; CPU path, not myelin |
| [`artifact-compatibility-plan.md`](./artifact-compatibility-plan.md) | Inventory IR plus `DryRunPlanner` coverage JSON |
| [`goz1-format.md`](./goz1-format.md) | GOZ1 container layout |

---

## Acceptance criteria

- [x] Repo documentation clearly states that CUDA kernel ownership lives in
      `myelin-accelerator`.
- [x] `grok-ozempic` has a backend integration plan (`BackendKernel` trait +
      dry-run planner).
- [x] Any local CUDA code is justified as Grok-specific glue, not reusable
      kernel infrastructure.
- [x] Future kernel work is routed to `myelin-accelerator` issues/PRs.
