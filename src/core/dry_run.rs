use std::collections::BTreeMap;

use crate::core::inventory::ModelInventory;
use crate::core::manifest::{DissectManifest, MANIFEST_NAME_CONVENTION_V2};
use crate::core::selection::TensorClass;
use crate::error::Result;
use crate::types::{QuantizationConfig, TensorPrecision};

/// A single planned backend kernel invocation derived from a manifest rule.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedKernelCall {
    /// Glob pattern or concrete tensor name from the manifest.
    pub matcher: String,
    /// The `BackendKernel` trait method that would execute this work.
    pub kernel_method: &'static str,
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
    /// How many tensors each backend method is planned to handle.
    pub by_method: BTreeMap<String, usize>,
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
    by_method: &mut BTreeMap<String, usize>,
    covered_by_rules: &mut usize,
) -> Result<()> {
    for entry in &manifest.preserve {
        let class = TensorClass::Preserve {
            reason: entry.reason.clone(),
        };
        let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
        let method = "convert_f32_to_f16_bytes";
        let estimated = estimate_tensor_count_for_manifest(inventory, manifest, &entry.name);
        rule_plans.push(PlannedKernelCall {
            matcher: entry.name.clone(),
            kernel_method: method,
            class,
            precision: TensorPrecision::Preserve,
            gif_threshold,
            estimated_tensor_count: estimated,
        });
        *by_method.entry(method.to_string()).or_insert(0) += estimated;
        *covered_by_rules += estimated;
    }
    Ok(())
}

fn plan_fp16_rules<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
    rule_plans: &mut Vec<PlannedKernelCall>,
    by_method: &mut BTreeMap<String, usize>,
    covered_by_rules: &mut usize,
) -> Result<()> {
    for entry in &manifest.fp16 {
        let class = TensorClass::Fp16 {
            reason: entry.reason.clone(),
        };
        let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
        let method = "convert_f32_to_f16_bytes";
        let estimated = estimate_tensor_count_for_manifest(inventory, manifest, &entry.name);
        rule_plans.push(PlannedKernelCall {
            matcher: entry.name.clone(),
            kernel_method: method,
            class,
            precision: TensorPrecision::Fp16,
            gif_threshold,
            estimated_tensor_count: estimated,
        });
        *by_method.entry(method.to_string()).or_insert(0) += estimated;
        *covered_by_rules += estimated;
    }
    Ok(())
}

fn plan_ternary_rules<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
    rule_plans: &mut Vec<PlannedKernelCall>,
    by_method: &mut BTreeMap<String, usize>,
    covered_by_rules: &mut usize,
) -> Result<()> {
    for entry in &manifest.ternary_candidates {
        let class = TensorClass::TernaryCandidate {
            rank: entry.rank,
            gif_threshold: entry.gif_threshold,
        };
        let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
        let method = if entry.name.contains("moe_expert") {
            "wrap_existing_int8_expert"
        } else if entry.name.contains("attn_proj_i8") {
            "wrap_existing_int8_unknown"
        } else {
            "quantize_f32"
        };
        let estimated = estimate_tensor_count_for_manifest(inventory, manifest, &entry.name);
        rule_plans.push(PlannedKernelCall {
            matcher: entry.name.clone(),
            kernel_method: method,
            class,
            precision: TensorPrecision::TernarySnN,
            gif_threshold,
            estimated_tensor_count: estimated,
        });
        *by_method.entry(method.to_string()).or_insert(0) += estimated;
        *covered_by_rules += estimated;
    }
    Ok(())
}

fn calculate_coverage(covered: usize, total: usize) -> CoverageStatus {
    if covered == total {
        CoverageStatus::Full
    } else if covered < total {
        CoverageStatus::Partial { missing: total - covered }
    } else {
        CoverageStatus::OverComplete { extra: covered - total }
    }
}

fn plan_default_rule<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
    rule_plans: &mut Vec<PlannedKernelCall>,
    by_method: &mut BTreeMap<String, usize>,
    covered_by_rules: &mut usize,
) -> Result<()> {
    let default_precision = manifest
        .defaults
        .precision
        .as_deref()
        .unwrap_or("ternary_snn");
    let default_class = TensorClass::Default;
    let (_precision, gif_threshold) = resolve_precision(&default_class, manifest, config)?;
    let default_method = match default_precision {
        "ternary_snn" => "quantize_f32",
        "fp16" => "convert_f32_to_f16_bytes",
        "preserve" => "convert_f32_to_f16_bytes",
        _ => "quantize_f32",
    };
    let explicit_covered: usize = rule_plans.iter().map(|p| p.estimated_tensor_count).sum();
    let inventory_total = inventory.total_tensors();
    let default_estimated = inventory_total.saturating_sub(explicit_covered);
    if default_estimated > 0 {
        rule_plans.push(PlannedKernelCall {
            matcher: "<defaults>".to_string(),
            kernel_method: default_method,
            class: default_class,
            precision: _precision,
            gif_threshold,
            estimated_tensor_count: default_estimated,
        });
        *by_method.entry(default_method.to_string()).or_insert(0) += default_estimated;
        *covered_by_rules += default_estimated;
    }
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

/// Plans which backend kernel calls each manifest rule would produce.
///
/// The planner reads the xai-dissect manifest, classifies every rule
/// (preserve / fp16 / ternary_candidates / defaults) through the existing
/// selection pipeline, and maps each to a `BackendKernel` method. The
/// result can be validated against the xai-dissect tensor inventory to
/// ensure full coverage.
pub struct DryRunPlanner;

impl DryRunPlanner {
    /// Walk every classification rule in the manifest and produce a
    /// `DryRunReport` mapping each rule to its planned backend kernel call.
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
        let mut by_method: BTreeMap<String, usize> = BTreeMap::new();
        let mut covered_by_rules = 0usize;

        plan_preserve_rules(
            inventory,
            manifest,
            config,
            &mut rule_plans,
            &mut by_method,
            &mut covered_by_rules,
        )?;
        plan_fp16_rules(
            inventory,
            manifest,
            config,
            &mut rule_plans,
            &mut by_method,
            &mut covered_by_rules,
        )?;
        plan_ternary_rules(
            inventory,
            manifest,
            config,
            &mut rule_plans,
            &mut by_method,
            &mut covered_by_rules,
        )?;
        plan_default_rule(
            inventory,
            manifest,
            config,
            &mut rule_plans,
            &mut by_method,
            &mut covered_by_rules,
        )?;

        let inventory_coverage = calculate_coverage(covered_by_rules, inventory.total_tensors());

        let backend_handled_total = by_method.values().sum();

        Ok(DryRunReport {
            rule_plans,
            coverage: CoverageSummary {
                by_method,
                covered_by_rules,
                inventory_total: inventory.total_tensors(),
                inventory_coverage,
            },
            backend_handled_total,
        })
    }

    /// Produce a machine-readable JSON mapping from rule matcher to planned
    /// backend method, suitable for comparison with xai-dissect artifacts.
    pub fn planned_backend_calls_json(
        report: &DryRunReport,
    ) -> BTreeMap<String, serde_json::Value> {
        let mut map = BTreeMap::new();
        for plan in &report.rule_plans {
            map.insert(
                plan.matcher.clone(),
                serde_json::json!({
                    "kernel_method": plan.kernel_method,
                    "precision": format!("{:?}", plan.precision),
                    "gif_threshold": plan.gif_threshold,
                    "estimated_tensor_count": plan.estimated_tensor_count,
                    "class": format!("{:?}", plan.class),
                }),
            );
        }
        map.insert(
            "__coverage__".to_string(),
            serde_json::json!({
                "by_method": report.coverage.by_method,
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
/// This legacy heuristic is retained only for backward compatibility with
/// older V1 manifests. Exact matching for V2 is always preferred.
fn estimate_tensor_count(pattern: &str) -> usize {
    // Wildcard patterns like "blk.*.ffn_up.weight" could match up to
    // GROK1_BLOCK_COUNT tensors (one per block).  Exact names count as 1.
    // This is the legacy V1 heuristic. For structural V2 manifests the
    // planner uses exact counts from Grok1Inventory instead (see
    // estimate_tensor_count_for_manifest).
    let star_count = pattern.matches('*').count();
    match star_count {
        0 => 1,
        _ => 8 * star_count, // rough heuristic: each wildcard ~8x
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
        estimate_tensor_count(pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::alignment::embedded_grok1_structural_manifest;
    use crate::core::alignment::plan_structural_manifest;
    use crate::core::grok1_inventory::Grok1Inventory;
    use crate::core::manifest::MANIFEST_NAME_CONVENTION_V2;
    use crate::types::GROK1_TENSOR_TOTAL;

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
    fn preserve_rules_map_to_convert_f32_to_f16_bytes() {
        let report = plan_structural_manifest();

        for plan in &report.rule_plans {
            if matches!(plan.class, TensorClass::Preserve { .. }) {
                assert_eq!(
                    plan.kernel_method, "convert_f32_to_f16_bytes",
                    "preserve rule '{}' should use convert_f32_to_f16_bytes, got {}",
                    plan.matcher, plan.kernel_method
                );
            }
        }
    }

    #[test]
    fn ternary_moe_expert_rules_map_to_wrap_int8() {
        let report = plan_structural_manifest();

        for plan in &report.rule_plans {
            if plan.matcher.contains("moe_expert") {
                assert_eq!(
                    plan.kernel_method, "wrap_existing_int8_expert",
                    "moe_expert rule '{}' should use wrap_existing_int8_expert, got {}",
                    plan.matcher, plan.kernel_method
                );
            }
        }
    }

    #[test]
    fn ternary_attn_proj_i8_rules_map_to_wrap_int8() {
        let report = plan_structural_manifest();

        for plan in &report.rule_plans {
            if plan.matcher.contains("attn_proj_i8") {
                assert_eq!(
                    plan.kernel_method, "wrap_existing_int8_unknown",
                    "attn_proj_i8 rule '{}' should use wrap_existing_int8_unknown, got {}",
                    plan.matcher, plan.kernel_method
                );
            }
        }
    }

    #[test]
    fn ternary_embedding_maps_to_quantize_f32() {
        let report = plan_structural_manifest();

        let embedding_plan = report
            .rule_plans
            .iter()
            .find(|p| p.matcher.contains("token_embedding"))
            .expect("embedding rule should exist");
        assert_eq!(
            embedding_plan.kernel_method, "quantize_f32",
            "embedding should use quantize_f32, got {}",
            embedding_plan.kernel_method
        );
    }

    #[test]
    fn default_rule_uses_manifest_default_precision() {
        let report = plan_structural_manifest();

        let default_plan = report.rule_plans.iter().find(|p| p.matcher == "<defaults>");
        if let Some(plan) = default_plan {
            assert!(
                plan.kernel_method == "quantize_f32"
                    || plan.kernel_method == "convert_f32_to_f16_bytes",
                "default rule should use quantize_f32 or convert_f32_to_f16_bytes, got {}",
                plan.kernel_method
            );
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
    fn by_method_sums_to_backend_handled_total() {
        let report = plan_structural_manifest();

        let sum: usize = report.coverage.by_method.values().sum();
        assert_eq!(
            sum, report.backend_handled_total,
            "by_method values should sum to backend_handled_total"
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
}
