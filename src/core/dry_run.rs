use std::collections::BTreeMap;

use crate::core::manifest::DissectManifest;
use crate::core::selection::TensorClass;
use crate::error::Result;
use crate::types::{GROK1_TENSOR_TOTAL, QuantizationConfig, TensorPrecision};

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
    /// When `blocks` are present in the manifest, the planner also accounts
    /// for per-block default tensors that fall through to the default
    /// precision tier.
    pub fn plan(manifest: &DissectManifest, config: &QuantizationConfig) -> Result<DryRunReport> {
        let mut rule_plans = Vec::new();
        let mut by_method: BTreeMap<String, usize> = BTreeMap::new();
        let mut covered_by_rules = 0usize;

        // 1. Preserve rules.
        // Note: `convert_f32_to_f16_bytes` is reported because the source
        // dtype may be F32 or BF16; `passthrough_f16` is only used when
        // the source is already FP16 (see `encode_fp16_bytes` in stream.rs).
        for entry in &manifest.preserve {
            let class = TensorClass::Preserve {
                reason: entry.reason.clone(),
            };
            let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
            let method = "convert_f32_to_f16_bytes";
            let estimated = estimate_tensor_count(&entry.name);
            rule_plans.push(PlannedKernelCall {
                matcher: entry.name.clone(),
                kernel_method: method,
                class,
                precision: TensorPrecision::Preserve,
                gif_threshold,
                estimated_tensor_count: estimated,
            });
            *by_method.entry(method.to_string()).or_insert(0) += estimated;
            covered_by_rules += estimated;
        }

        // 2. FP16 rules.
        // Same as preserve: source may be F32/BF16, so report the conversion
        // path rather than assuming FP16-at-rest.
        for entry in &manifest.fp16 {
            let class = TensorClass::Fp16 {
                reason: entry.reason.clone(),
            };
            let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
            let method = "convert_f32_to_f16_bytes";
            let estimated = estimate_tensor_count(&entry.name);
            rule_plans.push(PlannedKernelCall {
                matcher: entry.name.clone(),
                kernel_method: method,
                class,
                precision: TensorPrecision::Fp16,
                gif_threshold,
                estimated_tensor_count: estimated,
            });
            *by_method.entry(method.to_string()).or_insert(0) += estimated;
            covered_by_rules += estimated;
        }

        // 3. Ternary candidate rules.
        for entry in &manifest.ternary_candidates {
            let class = TensorClass::TernaryCandidate {
                rank: entry.rank,
                gif_threshold: entry.gif_threshold,
            };
            let (_precision, gif_threshold) = resolve_precision(&class, manifest, config)?;
            let method = "quantize_f32";
            let estimated = estimate_tensor_count(&entry.name);
            rule_plans.push(PlannedKernelCall {
                matcher: entry.name.clone(),
                kernel_method: method,
                class,
                precision: TensorPrecision::TernarySnN,
                gif_threshold,
                estimated_tensor_count: estimated,
            });
            *by_method.entry(method.to_string()).or_insert(0) += estimated;
            covered_by_rules += estimated;
        }

        // 4. Default rule — tensors not matched by any explicit list fall
        //    through to the manifest defaults or the pipeline default.
        let default_precision = manifest
            .defaults
            .precision
            .as_deref()
            .unwrap_or("ternary_snn");
        let default_class = TensorClass::Default;
        let (_precision, gif_threshold) = resolve_precision(&default_class, manifest, config)?;
        let default_method = match default_precision {
            "ternary_snn" => "quantize_f32",
            "fp16" => "passthrough_f16",
            "preserve" => "passthrough_f16",
            _ => "quantize_f32",
        };
        // Estimate default-covered tensors: total inventory minus explicit rules.
        let explicit_covered: usize = rule_plans.iter().map(|p| p.estimated_tensor_count).sum();
        let default_estimated = GROK1_TENSOR_TOTAL.saturating_sub(explicit_covered);
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
            covered_by_rules += default_estimated;
        }

        let inventory_total = GROK1_TENSOR_TOTAL;
        let inventory_coverage = if covered_by_rules == inventory_total {
            CoverageStatus::Full
        } else if covered_by_rules < inventory_total {
            CoverageStatus::Partial {
                missing: inventory_total - covered_by_rules,
            }
        } else {
            CoverageStatus::OverComplete {
                extra: covered_by_rules - inventory_total,
            }
        };

        let backend_handled_total = by_method.values().sum();

        Ok(DryRunReport {
            rule_plans,
            coverage: CoverageSummary {
                by_method,
                covered_by_rules,
                inventory_total,
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
/// matches in the Grok-1 inventory.
///
/// This is a best-effort estimate. Exact matching requires loading a real
/// checkpoint and running [`classify`](crate::core::selection::classify) on
/// every tensor name.
fn estimate_tensor_count(pattern: &str) -> usize {
    // Wildcard patterns like "blk.*.ffn_up.weight" could match up to
    // GROK1_BLOCK_COUNT tensors (one per block).  Exact names count as 1.
    let star_count = pattern.matches('*').count();
    match star_count {
        0 => 1,
        _ => 8 * star_count, // rough heuristic: each wildcard ~8x
    }
}
