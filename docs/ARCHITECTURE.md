# Architecture: kernel ownership boundary

This document defines the dependency boundary between `grok-ozempic` and
`myelin-accelerator` established by
[rmems/grok-ozempic#21](https://github.com/rmems/grok-ozempic/issues/21) /
[MET-101](https://linear.app/saaq-spiking-adaptive-activity/issue/MET-101).

## Principle

`grok-ozempic` is the **Grok-1-specific quantization and orchestration layer**.
It must not grow its own duplicated CUDA kernel stack unless a kernel is truly
Grok-specific. Kernel ownership lives in `myelin-accelerator` so that
binary/ternary/SAAQ kernels, bitpacking, benchmarks, and FFI remain reusable
across `corinth-canal`, Grok-1 experiments, and future Metis/Spikenaut work.

---

## Ownership table

| Area | Owner | Reason |
|------|-------|--------|
| Grok-1 checkpoint / shard handling | `grok-ozempic` | Grok-1 shard naming, safetensors layout |
| Tensor inventory and mapping | `grok-ozempic` | Manifest-driven precision classification |
| Router/expert-aware quantization planning | `grok-ozempic` | Grok-1 MoE structure |
| Per-expert quantization manifests | `grok-ozempic` | xai-dissect integration |
| Validation against xai-dissect artifacts | `grok-ozempic` | Grok-1 artifact contract |
| Dry-run quantization reports | `grok-ozempic` | Orchestration concern |
| High-level experiment orchestration | `grok-ozempic` | Pipeline entry points |
| GOZ1 binary container format | `grok-ozempic` | Grok-specific output format |
| Ternary bitpacking (`pack_trits`, `encode_trit`) | **`myelin-accelerator`** | Reusable across projects (moved in progress) |
| Binary / ternary bitpacking | `myelin-accelerator` | Generic CUDA kernel |
| Packed ternary GEMV / GEMM | `myelin-accelerator` | Generic CUDA kernel |
| SAAQ routing / reduction kernels | `myelin-accelerator` | Generic CUDA kernel |
| Kernel benchmarks | `myelin-accelerator` | Reusable infrastructure |
| Rust/CUDA FFI and launch helpers | `myelin-accelerator` | Reusable infrastructure |

---

## Backend integration layer

The `BackendKernel` trait in [`src/core/backend.rs`](../src/core/backend.rs)
defines the interface that `grok-ozempic` uses for all deployable kernel
operations.

```
grok-ozempic (orchestration)
    │
    ├── BackendKernel trait (src/core/backend.rs)
    │       │
    │       ├── LocalBackend       — delegates to quantizer.rs (CPU, current)
    │       └── MyelinBackend      — FFI to myelin-accelerator (future, stubbed)
    │
    ├── DryRunPlanner (src/core/dry_run.rs)
    │       — maps each tensor to its planned backend call
    │
    └── Existing: stream.rs, weight_pack.rs, manifest.rs, ...
```

### `LocalBackend` (current)

Wraps the existing Rust implementations in [`src/core/quantizer.rs`](../src/core/quantizer.rs).
Used by default. No behavior change; the same ternary packing functions execute
on CPU.

### `MyelinBackend` (future)

Stub that exists to establish the integration point. When `myelin-accelerator`
is added as a real dependency, the stub is replaced with an FFI bridge that
offloads kernel calls to CUDA.

---

## Dry-run planner

The `DryRunPlanner` reads the xai-dissect manifest and produces a
`DryRunReport` that maps each Grok-1 tensor to the backend kernel call it
would invoke. This serves two purposes:

1. **Validation** — the planned calls can be compared against the tensor
   inventory to catch misclassification or missing coverage.
2. **Backend readiness** — identifies which kernel operations a real backend
   must provide before the pipeline can switch from `LocalBackend`.

---

## What stays local (adapter/glue code)

Some tensor layout transforms are Grok-1-specific enough that they warrant
local glue code rather than backend kernel implementations:

| Transform | Justification |
|-----------|---------------|
| GOZ1 tensor table assembly | Grok-specific container format |
| Per-expert quantization manifest serialization | Grok-1 MoE structure |
| `ManifestEntry` → `PackTensorHeader` mapping | Combines manifest classification + GOZ1 format |
| Source dtype detection (F32/F16/BF16) | Input format handling, not kernel logic |

---

## Dependency status

`myelin-accelerator` is recorded as a **planned dependency** in
[`Cargo.toml`](../Cargo.toml) (commented out with a `#`). It becomes a real
dependency once the MyelinBackend FFI bridge is implemented.

---

## Acceptance criteria

- [x] Repo documentation clearly states that CUDA kernel ownership lives in
      `myelin-accelerator`.
- [x] `grok-ozempic` has a backend integration plan (`BackendKernel` trait +
      dry-run planner).
- [x] Any local CUDA code is justified as Grok-specific glue, not reusable
      kernel infrastructure.
- [x] Future kernel work is routed to `myelin-accelerator` issues/PRs.
