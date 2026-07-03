# Plan: Issues #16, #22, #27 — Testing, Alignment Verification, Docker CI

**Agent:** Kilo Code — OpenCode Go / Qwen3.7 Max
**Branch:** `testing-alignment-ci` (from `main`)
**Issues:** [#16](https://github.com/rmems/grok-ozempic/issues/16), [#22](https://github.com/rmems/grok-ozempic/issues/22), [#27](https://github.com/rmems/grok-ozempic/issues/27)

## Current State

- 89 tests pass, 0 failures
- `cargo clippy`, `cargo fmt --check`, `cargo doc` all clean
- Existing CI: `.github/workflows/rust.yml` (fmt/clippy/test/build/doc on GH Actions)
- No Docker files exist yet

---

## Phase 1: BackendKernel Error Handling (Issue #16)

### Task 1.1 — Add `BackendNotAvailable` error variant

**File:** `src/error.rs`

Add a new variant to `GrokOzempicError`:
```rust
#[error("backend not available: {0}")]
BackendNotAvailable(String),
```

### Task 1.2 — Change `BackendKernel` trait to return `Result`

**File:** `src/core/backend.rs`

Change all 5 trait methods from returning plain types to `Result<T, GrokOzempicError>`:
- `pack_ternary(&self, ternary: &[f32]) -> Result<Vec<u8>>`
- `quantize_f32(&self, weights: &[f32], gif_threshold: f32) -> Result<QuantizedTensor>`
- `quantize_f16(&self, weights: &[f16], gif_threshold: f32) -> Result<QuantizedTensor>`
- `passthrough_f16(&self, weights: &[f16]) -> Result<Vec<u8>>`
- `convert_f32_to_f16_bytes(&self, weights: &[f32]) -> Result<Vec<u8>>`

Add `use crate::error::Result;` import.

### Task 1.3 — Update `LocalBackend` impl

**File:** `src/core/backend.rs`

Wrap all return values in `Ok(...)`. No logic changes — delegates to `quantizer::*` as before.

### Task 1.4 — Replace `MyelinBackend` `unimplemented!()` with structured errors

**File:** `src/core/backend.rs`

Replace each `unimplemented!(...)` with:
```rust
Err(GrokOzempicError::BackendNotAvailable(
    "myelin-accelerator FFI not yet linked; use LocalBackend for CPU fallback".into()
))
```

### Task 1.5 — Fix any callers broken by trait change

Search for direct calls to `BackendKernel` methods. The `DryRunPlanner` references methods by string name only (`kernel_method: &'static str`), so no breakage there. Fix any other callers found.

### Validation (Phase 1)

```bash
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo doc --no-deps --all-features
```

---

## Phase 2: New Tests (Issue #16)

### Task 2.1 — `LocalBackend` unit tests

**File:** `src/core/backend.rs` (add `#[cfg(test)] mod tests`)

Tests for all 5 `BackendKernel` methods on `LocalBackend`:
1. `local_pack_ternary_round_trip` — pack known ternary values, verify byte layout matches `quantizer::pack_trits`
2. `local_quantize_f32_matches_direct_call` — compare `LocalBackend::quantize_f32` output to `quantizer::quantize_f32` (identical packed, rms, threshold, sparsity)
3. `local_quantize_f16_matches_direct_call` — same for f16 path
4. `local_passthrough_f16_matches_direct_call` — compare bytes
5. `local_convert_f32_to_f16_matches_direct_call` — compare bytes
6. `local_pack_ternary_empty_slice` — edge case: empty input
7. `local_quantize_f32_empty_slice` — edge case: empty input

### Task 2.2 — `MyelinBackend` error-path tests

**File:** `src/core/backend.rs` (same test module)

Tests for all 5 methods returning `Err(BackendNotAvailable)`:
1. `myelin_pack_ternary_returns_error`
2. `myelin_quantize_f32_returns_error`
3. `myelin_quantize_f16_returns_error`
4. `myelin_passthrough_f16_returns_error`
5. `myelin_convert_f32_to_f16_returns_error`

Each asserts `matches!(result, Err(GrokOzempicError::BackendNotAvailable(_)))`.

### Task 2.3 — `DryRunPlanner` kernel method mapping tests

**File:** `src/core/dry_run.rs` (extend existing `mod tests`)

1. `preserve_rules_map_to_convert_f32_to_f16_bytes` — verify all preserve rules produce `kernel_method == "convert_f32_to_f16_bytes"`
2. `ternary_moe_expert_rules_map_to_wrap_int8` — verify `moe_expert` ternary rules produce `wrap_existing_int8_expert`
3. `ternary_attn_proj_i8_rules_map_to_wrap_int8` — verify `attn_proj_i8` rules produce `wrap_existing_int8_unknown`
4. `ternary_embedding_maps_to_quantize_f32` — the token_embedding rule should map to `quantize_f32` (it's f32, not i8)
5. `default_rule_uses_manifest_default_precision` — verify the `<defaults>` rule uses the correct method for `ternary_snn`

### Task 2.4 — `CoverageSummary` tests

**File:** `src/core/dry_run.rs` (extend existing `mod tests`)

1. `coverage_full_when_rules_cover_all_770` — structural manifest produces `CoverageStatus::Full`
2. `coverage_partial_when_rules_miss_tensors` — construct a minimal manifest with fewer rules, verify `CoverageStatus::Partial { missing: N }`
3. `by_method_sums_to_backend_handled_total` — verify `by_method.values().sum() == backend_handled_total`
4. `planned_backend_calls_json_contains_coverage_key` — verify `__coverage__` key exists in JSON output

### Validation (Phase 2)

```bash
cargo test --all-targets --all-features
# Expect ~110+ tests passing
```

---

## Phase 3: Alignment Verification Tests (Issue #22)

### Task 3.1 — DryRunPlanner coverage vs inventory

**File:** `src/core/alignment.rs` (extend `mod tests`) or new test file

1. `dry_run_structural_manifest_coverage_is_full` — `DryRunPlanner::plan` with structural manifest produces `CoverageStatus::Full`
2. `dry_run_no_tensor_double_counted` — verify no two `PlannedKernelCall` entries match the same inventory tensor (run `glob_match` for each inventory tensor against all rule matchers; each tensor matches exactly one)
3. `dry_run_router_tensors_are_preserve` — all inventory tensors with `kind == "router"` are planned as `TensorClass::Preserve`
4. `dry_run_expert_tensors_are_ternary` — all `moe_expert.*` and `attn_proj_i8.*` inventory tensors are planned as `TensorClass::TernaryCandidate`
5. `dry_run_overcomplete_never_triggered_by_valid_manifest` — structural manifest never produces `CoverageStatus::OverComplete`
6. `every_manifest_rule_matches_at_least_one_inventory_tensor` — no orphan rules
7. `every_inventory_tensor_matched_by_manifest_rule` — no gaps; classify each tensor and verify it's not `Default`

### Task 3.2 — FP16/ternary/preserve boundary verification

**File:** `src/core/alignment.rs` (extend `mod tests`)

1. `preserve_fp16_ternary_boundaries_match_xai_dissect` — verify:
   - All 321 preserve tensors are `Preserve` class (64 routers + 256 block_norms + 1 final_norm)
   - All 449 ternary tensors are `TernaryCandidate` (192 MoE + 256 attn_proj_i8 + 1 embedding)
   - 0 fp16, 0 default
2. `no_router_tensor_classified_as_ternary` — invariant: routing-critical tensors never quantized

### Validation (Phase 3)

```bash
cargo test --all-targets --all-features
# All alignment tests pass; no discrepancies found
```

**Note:** If any discrepancies are found during implementation, document them as follow-up GitHub issues with specific tensor names and suggested fixes.

---

## Phase 4: Docker + GitHub Actions CI (Issue #27)

### Task 4.1 — Create Dockerfile

**File:** `Dockerfile` (project root)

Multi-stage build:
```dockerfile
# Stage 1: Build
FROM rust:1.85-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY dissect/ dissect/
RUN cargo build --release --all-features

# Stage 2: Test
FROM rust:1.85-slim AS tester
WORKDIR /app
COPY . .
RUN cargo test --all-targets --all-features && \
    cargo clippy --all-targets --all-features -- -D warnings && \
    cargo fmt --all -- --check

# Stage 3: Runtime (minimal)
FROM debian:bookworm-slim AS runtime
COPY --from=builder /app/target/release/grok-ozempic /usr/local/bin/
ENTRYPOINT ["grok-ozempic"]
```

### Task 4.2 — Create `.dockerignore`

**File:** `.dockerignore`

```
target/
.git/
.idea/
.beads/
*.md
```

### Task 4.3 — Create `docker-compose.yml`

**File:** `docker-compose.yml`

```yaml
services:
  test:
    build:
      context: .
      target: tester
  app:
    build:
      context: .
      target: runtime
```

### Task 4.4 — Enhance GitHub Actions workflow

**File:** `.github/workflows/rust.yml`

Add jobs:
1. **Docker build test** — build the Docker image on every PR/push, verify it builds successfully
2. **Security audit** — run `cargo audit` (add `actions-rs/audit-check` or equivalent)
3. **Coverage** — optional: add `cargo-tarpaulin` or `grcov` for test coverage reporting

### Validation (Phase 4)

```bash
docker build --target tester .
docker build --target runtime .
docker compose run test
```

---

## Execution Order

1. **Phase 1** (backend trait change) — must come first; changes the trait signature
2. **Phase 2** (new tests for #16) — depends on Phase 1
3. **Phase 3** (alignment tests for #22) — independent of Phase 1/2 but same test files
4. **Phase 4** (Docker + CI for #27) — independent; can be done in parallel with Phase 2/3

## Commit Strategy

- One commit per phase, or squash into a single PR commit
- Branch: `testing-alignment-ci`
- PR title: "test: full test coverage, alignment verification, Docker CI (#16, #22, #27)"

## Risks

- **Trait change breakage:** The `BackendKernel` trait is `pub` and re-exported. Any external consumers would break. Mitigation: no known external consumers; the crate is pre-1.0.
- **Alignment discrepancies:** If the structural manifest has gaps, tests will fail and need manifest fixes. Mitigation: existing alignment tests already show 0 mismatches.
- **Docker build time:** Rust compilation in Docker can be slow. Mitigation: multi-stage build with cargo cache.
