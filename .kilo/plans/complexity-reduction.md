# Plan: Issue #28 — Reduce Code Complexity (Real Structural Refactoring)

**Agent:** Kilo Code — xAI / Grok 4.3
**Branch:** `refactor-complexity-reduction`
**Issue:** [#28](https://github.com/rmems/grok-ozempic/issues/28)

## Problem Statement

The "test duplication" warnings from CodeRabbit/Codacy are symptoms of a deeper architectural problem:

1. **`grok1_inventory.rs` (288 lines)**: Hardcodes 770 `InventoryTensor` structs with massive repetition of `InventoryTensor { structural_name, expected_class, dtype, block, slot, kind }` patterns. No abstraction.

2. **`alignment.rs` (512 lines)**: ~200 lines of Grok-1-specific test code. The `AlignmentReport`, `TensorAlignment`, `check_alignment()` are all Grok-1 only. No trait.

3. **`dry_run.rs` (468 lines)**: `PlannedKernelCall`, `CoverageSummary`, `DryRunPlanner::plan()` are all Grok-1 specific. The `plan()` method walks `manifest.preserve`, `manifest.fp16`, `manifest.ternary_candidates` — hardcoded field names.

4. **No `ModelInventory` trait**: `Grok1Inventory` is a concrete struct. Adding Llama-3-MoE, Mixtral, etc. requires forking or massive conditional logic.

5. **Test helpers are Grok-1 only**: `plan_structural_manifest()`, `embedded_grok1_structural_manifest()` are hardcoded to Grok-1.

Extracting a 4-line helper (`plan_structural_manifest()`) does not reduce complexity. It papers over the symptom.

---

## Real Complexity Reduction Strategy

### Phase 1: Introduce `ModelInventory` Trait (Core Abstraction)

**File:** `src/core/inventory.rs` (new)

Define a trait that any model can implement:

```rust
pub trait ModelInventory {
    fn total_tensors(&self) -> usize;
    fn tensors(&self) -> &[InventoryTensor];
    fn count_by_expected_class(&self) -> (usize, usize, usize, usize); // preserve, fp16, ternary, default
    fn classify_tensor(&self, structural_name: &str) -> Option<TensorClass>;
}
```

`Grok1Inventory` becomes `impl ModelInventory for Grok1Inventory`.

**Benefit:** `alignment.rs` and `dry_run.rs` can be generic over `impl ModelInventory` instead of hardcoded to `Grok1Inventory`.

### Phase 2: Make `DryRunPlanner` Generic

**File:** `src/core/dry_run.rs`

Change:
```rust
pub fn plan(manifest: &DissectManifest, config: &QuantizationConfig) -> Result<DryRunReport>
```

To:
```rust
pub fn plan<I: ModelInventory>(inventory: &I, manifest: &DissectManifest, config: &QuantizationConfig) -> Result<DryRunReport>
```

The planner walks manifest rules and calls `inventory.classify_tensor(name)` instead of hardcoding Grok-1 logic.

**Benefit:** Same planner works for any model that implements `ModelInventory`.

### Phase 3: Make `check_alignment()` Generic

**File:** `src/core/alignment.rs`

Change:
```rust
pub fn check_alignment(inventory: &Grok1Inventory, manifest: &DissectManifest, config: &QuantizationConfig) -> AlignmentReport
```

To:
```rust
pub fn check_alignment<I: ModelInventory>(inventory: &I, manifest: &DissectManifest, config: &QuantizationConfig) -> AlignmentReport
```

**Benefit:** Alignment verification is now model-agnostic.

### Phase 4: Extract Model-Specific Data from Code

**File:** `src/core/grok1_inventory.rs`

Instead of 288 lines of `tensors.push(InventoryTensor { ... })`, load from a JSON/CSV data file at compile time or runtime.

**Benefit:** Reduces source LOC from 288 to ~20 (load + parse). Adding a new model is adding a data file, not code.

### Phase 5: Consolidate Test Helpers

**Files:** `src/core/alignment.rs`, `src/core/dry_run.rs`

Move `plan_structural_manifest()` and `embedded_grok1_structural_manifest()` into a shared test module (`src/core/test_fixtures.rs` or `#[cfg(test)] mod fixtures`).

**Benefit:** Single source of truth for test fixtures. Adding a new model's test fixtures is adding to the fixture module, not duplicating across test modules.

---

## Acceptance Criteria

- [ ] `ModelInventory` trait defined and `Grok1Inventory` implements it
- [ ] `DryRunPlanner::plan()` is generic over `impl ModelInventory`
- [ ] `check_alignment()` is generic over `impl ModelInventory`
- [ ] `grok1_inventory.rs` reduced from 288 LOC to <50 LOC (data-driven)
- [ ] Test helpers consolidated; no duplication of `plan_structural_manifest()` pattern
- [ ] All 118 tests pass with no behavior change
- [ ] `cargo clippy -- -D warnings` clean
- [ ] `cargo fmt -- --check` clean

## Non-Goals

- Do not implement support for Llama-3-MoE, Mixtral, etc. (that is issue #32)
- Do not change the GOZ1 artifact format
- Do not modify production streaming path (`stream.rs`)

## Risks

- **Trait design wrong:** If `ModelInventory` is poorly designed, generalization (issue #32) will be painful. Mitigation: keep trait minimal (4 methods).
- **Test behavior change:** Must verify 100% test parity after refactoring. Mitigation: run full test suite after every phase.

---

**Plan complete.** This is real complexity reduction, not cosmetic helper extraction.
