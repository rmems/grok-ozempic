# Kernel ownership boundary

`grok-ozempic` is the **Grok-1 quantization and orchestration** layer, not a generic CUDA stack.

| Area | Owner |
|------|--------|
| Checkpoint/shard handling, inventory, manifests, GOZ1 container | **grok-ozempic** |
| Ternary bitpacking kernels, packed GEMV/GEMM, SAAQ CUDA, FFI | **myelin-accelerator** |

- `BackendKernel` + `LocalBackend` (CPU) live in `src/core/backend.rs` / `quantizer.rs`.
- `MyelinBackend` is the future FFI seam; do not grow a parallel CUDA tree here.
- Full table: `docs/ARCHITECTURE.md`.

If a change is reusable kernel work, open/route it to `myelin-accelerator`, not this repo.
