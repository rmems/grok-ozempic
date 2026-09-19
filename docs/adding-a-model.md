# Adding a model family

The quantization **engine** (`DryRunPlanner`, `check_alignment`, `BackendKernel`,
`QuantizationConfig`) is model-agnostic. Grok-1 is the reference
[`ModelProfile`](../src/core/models/grok1.rs). A second in-tree family,
[`toy_moe`](../src/core/models/toy_moe.rs), proves a Mixtral-style inventory can
plan and align without forking the crate.

`magere-brug` is the registry / recipes / experiment-tracking layer. It should
**consume** these plugins, not reimplement packing. GOZ1 on-disk layout is
unchanged.

## What you implement

1. **Name convention** — add a `MANIFEST_NAME_CONVENTION_*` constant in
   [`src/core/manifest.rs`](../src/core/manifest.rs) and list it on
   `ACCEPTED_NAME_CONVENTIONS`. Unknown conventions still hard-fail at parse
   time. Legacy V1 (`blk.{L}.{role}.weight`) is the only convention that still
   uses the dry-run wildcard heuristic; everything else uses exact
   `ModelInventory::count_matching`.
2. **Inventory** — a `VecInventory` (or a dedicated type implementing
   `ModelInventory`) listing every tensor name, expected class, dtype, and
   optional block/slot.
3. **`ModelProfile`** — associated type `Inventory: ModelInventory`, plus
   `family()`, `source()`, `tensor_name_convention()`, `inventory()`,
   `manifest()`, and optional `default_gif_threshold()`.
4. **Manifest** — a dissect-schema JSON (schema v1) with preserve / fp16 /
   ternary globs in **this family's names**. Prefer `parse_manifest_bytes` so
   the loader contract is tested. Classification remains
   preserve > fp16 > ternary_candidates > defaults.
5. **Tests in the plugin module** — not copy-pasted into `dry_run.rs` /
   `alignment.rs`. Use the crate's test harness:

   ```rust
   let report = crate::core::test_support::plan_profile(&MyProfile);
   let aligned = crate::core::test_support::align_profile(&MyProfile);
   assert_eq!(report.coverage.inventory_coverage, CoverageStatus::Full);
   assert!(aligned.is_aligned());
   ```

   Shared helpers live in `src/core/test_support.rs` (cfg(test) only).

## What you do not change

- GOZ1 `weight_pack` / `TENSOR_F16` / `TENSOR_TERNARY`
- CUDA kernels (those belong in `myelin-accelerator`)
- A full model registry or experiment tracker (belongs in `magere-brug`)
- Grok-1 packing CLI defaults (`append_grok1_arch_metadata`,
  `use_embedded_baseline`) until a follow-up adds a generic arch-metadata hook

`run_quantization` still writes Grok-1 architecture keys into GOZ1 metadata.
Unmatched tensor names **fail closed** for every non-V1 convention (same
`ManifestV2UnmatchedTensor` error as structural V2). V1 `blk.*` keeps
defaults fallthrough. Dry-run matches packing: a fail-closed manifest
with unmatched inventory tensors reports `CoverageStatus::Partial` and
does not synthesize a `<defaults>` plan. Complete plugins still need
`CoverageStatus::Full` via explicit rules.

## Classification

Default: `DissectManifest` implements `TensorClassifier` via segment-anchored
globs (`selection::classify`). If a family cannot be expressed as dissect
globs, implement `TensorClassifier` and call `check_alignment_with` — do not
fork `check_alignment`.

## Checklist

- [ ] Convention registered; a fixture manifest parses
- [ ] Inventory names are unique; class counts match the layout
- [ ] `plan_profile` reports `CoverageStatus::Full`
- [ ] `align_profile` is aligned (routers/norms never leak to ternary)
- [ ] GIF / precision defaults come from the profile or the manifest, not
      hardcoded Grok-1 constants
- [ ] No new 3-line `inventory + manifest + QuantizationConfig::default()`
      setup in core test modules — use the harness
