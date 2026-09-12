use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::core::inventory::ModelInventory;
use crate::core::manifest::{DissectManifest, MANIFEST_NAME_CONVENTION_V2};
use crate::core::selection::TensorClass;
use crate::error::{GrokOzempicError, Result};
use crate::types::{QuantizationConfig, TensorPrecision};

/// Orchestration-level verb for a planned tensor transform.
///
/// This is not a 1:1 map of [`crate::core::backend::BackendKernel`] methods:
/// wrapping an already-quantized source is an artifact-path operation, not a
/// kernel. Convert and quantize map onto real `BackendKernel` methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    /// Ternary-quantize floating-point source weights
    /// ([`crate::core::backend::BackendKernel::quantize_f32`]).
    QuantizeTernary,
    /// Convert floating-point source to FP16 bytes
    /// ([`crate::core::backend::BackendKernel::convert_f32_to_f16_bytes`]).
    ConvertFp16,
    /// Wrap an already-quantized (typically int8) payload without re-quantizing.
    WrapExistingQuantized,
}

impl OperationKind {
    /// Manifest / JSON / artifact-policy spelling of this verb.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QuantizeTernary => "quantize_ternary",
            Self::ConvertFp16 => "convert_fp16",
            Self::WrapExistingQuantized => "wrap_existing_quantized",
        }
    }

    /// Wrap vs re-quantize from a single source dtype.
    ///
    /// Unknown dtypes fail closed rather than guessing from a glob substring.
    pub fn for_ternary_source_dtype(dtype: &str) -> Result<Self> {
        if is_already_quantized_dtype(dtype) {
            Ok(Self::WrapExistingQuantized)
        } else if is_float_source_dtype(dtype) {
            Ok(Self::QuantizeTernary)
        } else {
            Err(GrokOzempicError::MixedInventoryDtype {
                pattern: format!("dtype={dtype}"),
            })
        }
    }
}

impl std::fmt::Display for OperationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// True when the source is already a quantized integer payload (wrap, don't
/// re-quantize). Inventory uses `i8`; the artifact planner uses `int8`.
pub fn is_already_quantized_dtype(dtype: &str) -> bool {
    matches!(dtype, "i8" | "int8" | "u8")
}

fn is_float_source_dtype(dtype: &str) -> bool {
    matches!(dtype, "f32" | "f16" | "bf16" | "fp16")
}

/// A single planned backend kernel invocation derived from a manifest rule.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedKernelCall {
    /// Glob pattern or concrete tensor name from the manifest.
    pub matcher: String,
    /// Orchestration verb this rule would execute.
    pub operation: OperationKind,
    /// Classification outcome.
    pub class: TensorClass,
    /// Resolved precision.
    pub precision: TensorPrecision,
    /// Effective GIF threshold (meaningful for ternary).
    pub gif_threshold: f32,
    /// Estimated tensor count this rule covers (based on Grok-1 inventory).
    pub estimated_tensor_count: usize,
}

/// Coverage analysis against the known Grok-1 tensor inventory.
#[derive(Debug, Clone)]
pub struct CoverageSummary {
    /// How many tensors each operation is planned to handle.
    pub by_operation: BTreeMap<OperationKind, usize>,
    /// Total tensors covered by manifest rules.
    pub covered_by_rules: usize,
    /// Total tensors in the Grok-1 baseline inventory.
    pub inventory_total: usize,
    /// Match status.
    pub inventory_coverage: CoverageStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoverageStatus {
    Full,
    Partial { missing: usize },
    OverComplete { extra: usize },
}

fn plan_preserve_rules<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
    rule_plans: &mut Vec<PlannedKernelCall>,
    by_operation: &mut BTreeMap<OperationKind, usize>,
    covered_by_rules: &mut usize,
) -> Result<()> {
    for entry in &manifest.preserve {
        let class = TensorClass::Preserve {
            reason: entry.reason.clone(),
        };
        let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
        let operation = OperationKind::ConvertFp16;
        let estimated = estimate_tensor_count_for_manifest(inventory, manifest, &entry.name);
        rule_plans.push(PlannedKernelCall {
            matcher: entry.name.clone(),
            operation,
            class,
            precision: TensorPrecision::Preserve,
            gif_threshold,
            estimated_tensor_count: estimated,
        });
        *by_operation.entry(operation).or_insert(0) += estimated;
        *covered_by_rules += estimated;
    }
    Ok(())
}

fn plan_fp16_rules<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
    rule_plans: &mut Vec<PlannedKernelCall>,
    by_operation: &mut BTreeMap<OperationKind, usize>,
    covered_by_rules: &mut usize,
) -> Result<()> {
    for entry in &manifest.fp16 {
        let class = TensorClass::Fp16 {
            reason: entry.reason.clone(),
        };
        let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
        let operation = OperationKind::ConvertFp16;
        let estimated = estimate_tensor_count_for_manifest(inventory, manifest, &entry.name);
        rule_plans.push(PlannedKernelCall {
            matcher: entry.name.clone(),
            operation,
            class,
            precision: TensorPrecision::Fp16,
            gif_threshold,
            estimated_tensor_count: estimated,
        });
        *by_operation.entry(operation).or_insert(0) += estimated;
        *covered_by_rules += estimated;
    }
    Ok(())
}

fn plan_ternary_rules<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
    rule_plans: &mut Vec<PlannedKernelCall>,
    by_operation: &mut BTreeMap<OperationKind, usize>,
    covered_by_rules: &mut usize,
) -> Result<()> {
    for entry in &manifest.ternary_candidates {
        let class = TensorClass::TernaryCandidate {
            rank: entry.rank,
            gif_threshold: entry.gif_threshold,
        };
        let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
        let operation = ternary_operation_from_inventory(inventory, &entry.name)?;
        let estimated = estimate_tensor_count_for_manifest(inventory, manifest, &entry.name);
        rule_plans.push(PlannedKernelCall {
            matcher: entry.name.clone(),
            operation,
            class,
            precision: TensorPrecision::TernarySnn,
            gif_threshold,
            estimated_tensor_count: estimated,
        });
        *by_operation.entry(operation).or_insert(0) += estimated;
        *covered_by_rules += estimated;
    }
    Ok(())
}

/// Wrap vs re-quantize from matching inventory dtypes, never from a glob
/// substring. No matches (legacy V1 names against a structural inventory)
/// default to quantize, which is the float-stream path.
fn ternary_operation_from_inventory<I: ModelInventory>(
    inventory: &I,
    pattern: &str,
) -> Result<OperationKind> {
    let mut saw_quantized = false;
    let mut saw_float = false;
    let mut saw_other = false;
    for tensor in inventory.tensors() {
        if !crate::core::selection::glob_match(pattern, &tensor.structural_name) {
            continue;
        }
        if is_already_quantized_dtype(tensor.dtype) {
            saw_quantized = true;
        } else if is_float_source_dtype(tensor.dtype) {
            saw_float = true;
        } else {
            saw_other = true;
        }
    }
    match (saw_quantized, saw_float, saw_other) {
        (true, false, false) => Ok(OperationKind::WrapExistingQuantized),
        (false, true, false) | (false, false, false) => Ok(OperationKind::QuantizeTernary),
        _ => Err(GrokOzempicError::MixedInventoryDtype {
            pattern: pattern.to_string(),
        }),
    }
}

fn calculate_coverage(covered: usize, total: usize) -> CoverageStatus {
    if covered == total {
        CoverageStatus::Full
    } else if covered < total {
        CoverageStatus::Partial {
            missing: total - covered,
        }
    } else {
        CoverageStatus::OverComplete {
            extra: covered - total,
        }
    }
}

fn plan_default_rule<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
    rule_plans: &mut Vec<PlannedKernelCall>,
    by_operation: &mut BTreeMap<OperationKind, usize>,
    covered_by_rules: &mut usize,
) -> Result<()> {
    let inventory_total = inventory.total_tensors();
    let default_estimated = inventory_total.saturating_sub(*covered_by_rules);
    if default_estimated == 0 {
        return Ok(());
    }
    let default_class = TensorClass::Default;
    let (precision, gif_threshold) = resolve_precision(&default_class, manifest, config)?;
    let operation = match precision {
        TensorPrecision::TernarySnn => OperationKind::QuantizeTernary,
        TensorPrecision::Fp16 | TensorPrecision::Preserve => OperationKind::ConvertFp16,
    };
    rule_plans.push(PlannedKernelCall {
        matcher: "<defaults>".to_string(),
        operation,
        class: default_class,
        precision,
        gif_threshold,
        estimated_tensor_count: default_estimated,
    });
    *by_operation.entry(operation).or_insert(0) += default_estimated;
    *covered_by_rules += default_estimated;
    Ok(())
}

/// Top-level dry-run output.
#[derive(Debug, Clone)]
pub struct DryRunReport {
    /// Per-rule planned kernel calls.
    pub rule_plans: Vec<PlannedKernelCall>,
    /// Aggregate coverage analysis.
    pub coverage: CoverageSummary,
    /// Projected backend-handled tensor count.
    pub backend_handled_total: usize,
}

impl DryRunReport {
    pub fn summary(&self) -> String {
        format!(
            "DryRunReport: {rules} rules, {total} tensors planned, \
             {handled} backend-handled, coverage={cov:?}",
            rules = self.rule_plans.len(),
            total = self.coverage.covered_by_rules,
            handled = self.backend_handled_total,
            cov = self.coverage.inventory_coverage,
        )
    }
}

/// Plans which orchestration verbs each manifest rule would produce.
///
/// The planner reads the xai-dissect manifest, classifies every rule
/// (preserve / fp16 / ternary_candidates / defaults) through the existing
/// selection pipeline, and maps each to an [`OperationKind`]. Wrap vs
/// re-quantize is decided from inventory dtype, not glob substrings.
/// The result can be validated against the xai-dissect tensor inventory
/// to ensure full coverage.
pub struct DryRunPlanner;

impl DryRunPlanner {
    /// Walk every classification rule in the manifest and produce a
    /// `DryRunReport` mapping each rule to its planned operation.
    ///
    /// The `inventory` parameter provides model-specific tensor counts for
    /// accurate per-rule estimates. For V2 structural manifests, counts are
    /// taken exactly from the inventory; for legacy V1 manifests, a heuristic
    /// is used.
    pub fn plan<I: ModelInventory>(
        inventory: &I,
        manifest: &DissectManifest,
        config: &QuantizationConfig,
    ) -> Result<DryRunReport> {
        let mut rule_plans = Vec::new();
        let mut by_operation: BTreeMap<OperationKind, usize> = BTreeMap::new();
        let mut covered_by_rules = 0usize;

        Self::plan_all_rules(
            inventory,
            manifest,
            config,
            &mut rule_plans,
            &mut by_operation,
            &mut covered_by_rules,
        )?;

        let inventory_coverage = calculate_coverage(covered_by_rules, inventory.total_tensors());
        let backend_handled_total = by_operation.values().sum();

        Ok(DryRunReport {
            rule_plans,
            coverage: CoverageSummary {
                by_operation,
                covered_by_rules,
                inventory_total: inventory.total_tensors(),
                inventory_coverage,
            },
            backend_handled_total,
        })
    }

    fn plan_all_rules<I: ModelInventory>(
        inventory: &I,
        manifest: &DissectManifest,
        config: &QuantizationConfig,
        rule_plans: &mut Vec<PlannedKernelCall>,
        by_operation: &mut BTreeMap<OperationKind, usize>,
        covered_by_rules: &mut usize,
    ) -> Result<()> {
        plan_preserve_rules(
            inventory,
            manifest,
            config,
            rule_plans,
            by_operation,
            covered_by_rules,
        )?;
        plan_fp16_rules(
            inventory,
            manifest,
            config,
            rule_plans,
            by_operation,
            covered_by_rules,
        )?;
        plan_ternary_rules(
            inventory,
            manifest,
            config,
            rule_plans,
            by_operation,
            covered_by_rules,
        )?;
        plan_default_rule(
            inventory,
            manifest,
            config,
            rule_plans,
            by_operation,
            covered_by_rules,
        )
    }

    /// Produce a machine-readable JSON mapping from rule matcher to planned
    /// operation, suitable for comparison with xai-dissect artifacts.
    pub fn planned_backend_calls_json(
        report: &DryRunReport,
    ) -> BTreeMap<String, serde_json::Value> {
        let mut map = BTreeMap::new();
        for plan in &report.rule_plans {
            map.insert(
                plan.matcher.clone(),
                serde_json::json!({
                    "operation": plan.operation,
                    "precision": plan.precision,
                    "gif_threshold": plan.gif_threshold,
                    "estimated_tensor_count": plan.estimated_tensor_count,
                    "class": format!("{:?}", plan.class),
                }),
            );
        }
        map.insert(
            "__coverage__".to_string(),
            serde_json::json!({
                "by_operation": report.coverage.by_operation,
                "covered_by_rules": report.coverage.covered_by_rules,
                "inventory_total": report.coverage.inventory_total,
                "coverage": format!("{:?}", report.coverage.inventory_coverage),
                "backend_handled_total": report.backend_handled_total,
            }),
        );
        map
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_precision(
    class: &TensorClass,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
) -> Result<(TensorPrecision, f32)> {
    crate::core::precision::decide(class, Some(manifest), config)
}

/// Heuristically estimate how many concrete tensors a single glob pattern
/// matches in the Grok-1 inventory (legacy V1 `blk.*` naming convention).
///
/// For the xai-dissect structural manifest (V2 `block_*.slot_*` convention)
/// the planner uses exact counts from [`ModelInventory::count_matching`]
/// instead, so that dry-run coverage reports are accurate for the 770-tensor
/// inventory (e.g. 64 for `block_*.slot_11.router`).
///
fn estimate_tensor_count<I: ModelInventory>(inventory: &I, pattern: &str) -> usize {
    // Wildcard patterns like "blk.*.ffn_up.weight" could match up to
    // GROK1_BLOCK_COUNT tensors (one per block).  Exact names count as 1.
    // This is the legacy V1 heuristic. For structural V2 manifests the
    // planner uses exact counts from Grok1Inventory instead (see
    // estimate_tensor_count_for_manifest).
    let star_count = pattern.matches('*').count();
    match star_count {
        0 => 1,
        _ => {
            // Scale the wildcard multiplier dynamically based on the number of blocks in the inventory.
            // For Grok-1 (64 blocks), this results in 64 / 8 = 8, matching the legacy heuristic.
            let mut unique_blocks = std::collections::HashSet::new();
            for t in inventory.tensors() {
                if let Some(b) = t.block {
                    unique_blocks.insert(b);
                }
            }
            let num_blocks = unique_blocks.len();
            let multiplier = if num_blocks > 0 {
                (num_blocks / 8).max(1)
            } else {
                8
            };
            multiplier * star_count
        }
    }
}

fn estimate_tensor_count_for_manifest<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    pattern: &str,
) -> usize {
    if manifest.model.tensor_name_convention == MANIFEST_NAME_CONVENTION_V2 {
        inventory.count_matching(pattern)
    } else {
        estimate_tensor_count(inventory, pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::alignment::embedded_grok1_structural_manifest;
    use crate::core::alignment::plan_structural_manifest;
    use crate::core::grok1_inventory::Grok1Inventory;
    use crate::core::inventory::{InventoryTensor, ModelInventory};
    use crate::core::manifest::MANIFEST_NAME_CONVENTION_V2;
    use crate::types::{GROK1_TENSOR_TOTAL, QuantizationConfig};

    #[test]
    fn structural_manifest_router_rule_counts_exactly_64() {
        let m = embedded_grok1_structural_manifest();
        assert_eq!(m.model.tensor_name_convention, MANIFEST_NAME_CONVENTION_V2);

        // The single router rule must count 64 (one per block), not the legacy 8.
        let router_rule = m
            .preserve
            .iter()
            .find(|e| e.name.contains("router"))
            .expect("structural manifest has router preserve rule");
        let count =
            estimate_tensor_count_for_manifest(&Grok1Inventory::full(), m, &router_rule.name);
        assert_eq!(
            count, 64,
            "router rule should count 64 via inventory, got {count}"
        );
    }

    #[test]
    fn structural_manifest_dry_run_covers_all_770_or_reports_reasonable_default() {
        let report = plan_structural_manifest();

        // With exact counts, covered_by_rules should be much closer to 770 than the old
        // heuristic (which produced ~8 per rule + 672+ in <defaults>).
        let covered = report.coverage.covered_by_rules;
        assert!(
            covered >= 700,
            "structural manifest dry-run should cover most of 770 via exact globs, got {covered}"
        );
        assert_eq!(report.coverage.inventory_total, GROK1_TENSOR_TOTAL);
    }

    #[test]
    fn preserve_rules_map_to_convert_fp16() {
        let report = plan_structural_manifest();

        for plan in &report.rule_plans {
            if matches!(plan.class, TensorClass::Preserve { .. }) {
                assert_eq!(
                    plan.operation,
                    OperationKind::ConvertFp16,
                    "preserve rule '{}' should use ConvertFp16, got {}",
                    plan.matcher,
                    plan.operation
                );
            }
        }
    }

    #[test]
    fn ternary_i8_inventory_rules_map_to_wrap() {
        let report = plan_structural_manifest();

        for plan in &report.rule_plans {
            if plan.matcher.contains("moe_expert") || plan.matcher.contains("attn_proj_i8") {
                assert_eq!(
                    plan.operation,
                    OperationKind::WrapExistingQuantized,
                    "i8 ternary rule '{}' should wrap, got {}",
                    plan.matcher,
                    plan.operation
                );
            }
        }
    }

    #[test]
    fn ternary_embedding_maps_to_quantize_ternary() {
        let report = plan_structural_manifest();

        let embedding_plan = report
            .rule_plans
            .iter()
            .find(|p| p.matcher.contains("token_embedding"))
            .expect("embedding rule should exist");
        assert_eq!(
            embedding_plan.operation,
            OperationKind::QuantizeTernary,
            "embedding should use QuantizeTernary, got {}",
            embedding_plan.operation
        );
    }

    #[test]
    fn default_rule_uses_manifest_default_precision() {
        let report = plan_structural_manifest();

        let default_plan = report.rule_plans.iter().find(|p| p.matcher == "<defaults>");
        if let Some(plan) = default_plan {
            match plan.operation {
                OperationKind::QuantizeTernary | OperationKind::ConvertFp16 => {}
                other => {
                    panic!("default rule should use QuantizeTernary or ConvertFp16, got {other}")
                }
            }
        } else {
            assert_eq!(
                report.coverage.inventory_coverage,
                CoverageStatus::Full,
                "no <defaults> rule should only be absent when coverage is Full"
            );
        }
    }

    #[test]
    fn coverage_full_when_rules_cover_all_770() {
        let report = plan_structural_manifest();

        assert_eq!(
            report.coverage.inventory_coverage,
            CoverageStatus::Full,
            "structural manifest should produce CoverageStatus::Full"
        );
    }

    #[test]
    fn by_operation_sums_to_backend_handled_total() {
        let report = plan_structural_manifest();

        let sum: usize = report.coverage.by_operation.values().sum();
        assert_eq!(
            sum, report.backend_handled_total,
            "by_operation values should sum to backend_handled_total"
        );
    }

    #[test]
    fn planned_backend_calls_json_contains_coverage_key() {
        let report = plan_structural_manifest();
        let json = DryRunPlanner::planned_backend_calls_json(&report);

        assert!(
            json.contains_key("__coverage__"),
            "JSON output should contain __coverage__ key"
        );
    }

    #[test]
    fn planned_backend_calls_json_uses_serde_precision_wire_form() {
        let report = plan_structural_manifest();
        let json = DryRunPlanner::planned_backend_calls_json(&report);
        let embedding = json
            .iter()
            .find(|(matcher, _)| matcher.contains("token_embedding"))
            .map(|(_, value)| value)
            .expect("embedding rule should exist");
        assert_eq!(
            embedding["precision"], "ternary_snn",
            "dry-run JSON must emit the serde wire form, not Debug/PascalCase"
        );
        assert_eq!(embedding["operation"], "quantize_ternary");
    }

    #[test]
    fn every_planned_operation_is_exhaustively_known() {
        let report = plan_structural_manifest();
        for plan in &report.rule_plans {
            match plan.operation {
                OperationKind::QuantizeTernary
                | OperationKind::ConvertFp16
                | OperationKind::WrapExistingQuantized => {}
            }
        }
    }

    struct TinyInv(Vec<InventoryTensor>);

    impl ModelInventory for TinyInv {
        fn total_tensors(&self) -> usize {
            self.0.len()
        }
        fn tensors(&self) -> &[InventoryTensor] {
            &self.0
        }
    }

    fn tiny_tensor(name: &str, dtype: &'static str) -> InventoryTensor {
        InventoryTensor {
            structural_name: name.into(),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype,
            block: None,
            slot: None,
            kind: "test",
        }
    }

    fn v2_manifest_with_ternary(pattern: &str) -> DissectManifest {
        use crate::core::manifest::{
            MANIFEST_SCHEMA_VERSION, ManifestDefaults, ManifestModel, TernaryCandidate,
        };
        DissectManifest {
            schema: "xai-dissect.manifest".into(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            model: ManifestModel {
                family: "grok-1".into(),
                source: "xai-org/grok-1".into(),
                tensor_name_convention: MANIFEST_NAME_CONVENTION_V2.into(),
            },
            produced_by: None,
            defaults: ManifestDefaults {
                precision: Some("ternary_snn".into()),
                gif_threshold: None,
            },
            preserve: vec![],
            fp16: vec![],
            ternary_candidates: vec![TernaryCandidate {
                name: pattern.into(),
                rank: None,
                gif_threshold: None,
            }],
            blocks: vec![],
        }
    }

    #[test]
    fn ternary_i8_source_plans_wrap_even_without_moe_expert_in_glob() {
        let inv = TinyInv(vec![tiny_tensor("block_000.slot_00.already_int8", "i8")]);
        let m = v2_manifest_with_ternary("block_*.slot_00.already_int8");
        let report = DryRunPlanner::plan(&inv, &m, &QuantizationConfig::default())
            .expect("plan should succeed");
        let plan = report
            .rule_plans
            .iter()
            .find(|p| p.matcher.contains("already_int8"))
            .expect("ternary rule");
        assert_eq!(plan.operation, OperationKind::WrapExistingQuantized);
    }

    #[test]
    fn glob_containing_moe_expert_does_not_force_wrap_on_f32() {
        let inv = TinyInv(vec![tiny_tensor(
            "block_000.slot_00.moe_expert.floaty",
            "f32",
        )]);
        let m = v2_manifest_with_ternary("block_*.slot_00.moe_expert.floaty");
        let report = DryRunPlanner::plan(&inv, &m, &QuantizationConfig::default())
            .expect("plan should succeed");
        let plan = report
            .rule_plans
            .iter()
            .find(|p| p.matcher.contains("moe_expert"))
            .expect("ternary rule");
        assert_eq!(
            plan.operation,
            OperationKind::QuantizeTernary,
            "f32 source must quantize even if the glob contains moe_expert"
        );
    }

    #[test]
    fn mixed_inventory_dtypes_fail_closed() {
        let inv = TinyInv(vec![
            tiny_tensor("block_000.slot_00.mixed", "i8"),
            tiny_tensor("block_001.slot_00.mixed", "f32"),
        ]);
        let m = v2_manifest_with_ternary("block_*.slot_00.mixed");
        let err = DryRunPlanner::plan(&inv, &m, &QuantizationConfig::default())
            .expect_err("mixed dtypes must fail closed");
        assert!(
            matches!(
                err,
                crate::error::GrokOzempicError::MixedInventoryDtype { .. }
            ),
            "got {err:?}"
        );
    }
}
