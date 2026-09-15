use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::core::inventory::ModelInventory;
use crate::core::manifest::{DissectManifest, parse_manifest_bytes};
use crate::core::selection::{TensorClass, TensorClassifier, classify};
use crate::types::QuantizationConfig;

pub const GROK1_STRUCTURAL_MANIFEST_JSON: &str =
    include_str!("../../dissect/grok-1/structural-manifest.json");

pub fn embedded_grok1_structural_manifest() -> &'static DissectManifest {
    static CACHE: OnceLock<DissectManifest> = OnceLock::new();
    CACHE.get_or_init(|| {
        parse_manifest_bytes(
            GROK1_STRUCTURAL_MANIFEST_JSON.as_bytes(),
            "<embedded grok-1 structural manifest>",
        )
        .expect("embedded structural manifest must parse")
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClassMatch {
    Match,
    Mismatch {
        got: TensorClass,
        expected: TensorClass,
    },
}

#[derive(Debug, Clone)]
pub struct TensorAlignment {
    pub structural_name: String,
    pub expected_class: TensorClass,
    pub actual_class: TensorClass,
    pub match_status: ClassMatch,
}

#[derive(Debug, Clone)]
pub struct AlignmentReport {
    pub total_inventory_tensors: usize,
    pub matched: usize,
    pub mismatched: usize,
    pub preserve_expected_count: usize,
    pub fp16_expected_count: usize,
    pub ternary_expected_count: usize,
    pub default_expected_count: usize,
    pub preserve_actual_count: usize,
    pub fp16_actual_count: usize,
    pub ternary_actual_count: usize,
    pub default_actual_count: usize,
    pub mismatches: Vec<TensorAlignment>,
    pub boundary_summary: BTreeMap<String, usize>,
}

impl AlignmentReport {
    pub fn is_aligned(&self) -> bool {
        self.mismatches.is_empty()
    }

    pub fn summary(&self) -> String {
        let status = if self.is_aligned() {
            "ALIGNED"
        } else {
            "MISALIGNED"
        };
        format!(
            "AlignmentReport: {status} | {total} tensors, {matched} matched, {mismatched} mismatched \
             | expected: {preserve_exp}P/{fp16_exp}F/{ternary_exp}T/{default_exp}D \
             | actual: {preserve_act}P/{fp16_act}F/{ternary_act}T/{default_act}D",
            total = self.total_inventory_tensors,
            matched = self.matched,
            mismatched = self.mismatched,
            preserve_exp = self.preserve_expected_count,
            fp16_exp = self.fp16_expected_count,
            ternary_exp = self.ternary_expected_count,
            default_exp = self.default_expected_count,
            preserve_act = self.preserve_actual_count,
            fp16_act = self.fp16_actual_count,
            ternary_act = self.ternary_actual_count,
            default_act = self.default_actual_count,
        )
    }
}

#[derive(Default)]
struct ClassCounts {
    preserve: usize,
    fp16: usize,
    ternary: usize,
    default: usize,
}

impl ClassCounts {
    fn bump(&mut self, class: &TensorClass) {
        match class {
            TensorClass::Preserve { .. } => self.preserve += 1,
            TensorClass::Fp16 { .. } => self.fp16 += 1,
            TensorClass::TernaryCandidate { .. } => self.ternary += 1,
            TensorClass::Default => self.default += 1,
        }
    }
}

fn record_mismatch(
    mismatches: &mut Vec<TensorAlignment>,
    boundary_summary: &mut BTreeMap<String, usize>,
    structural_name: String,
    expected: TensorClass,
    actual: TensorClass,
) {
    let boundary_key = format!("{expected:?} -> {actual:?}");
    *boundary_summary.entry(boundary_key).or_insert(0) += 1;
    mismatches.push(TensorAlignment {
        structural_name,
        expected_class: expected.clone(),
        actual_class: actual.clone(),
        match_status: ClassMatch::Mismatch {
            got: actual,
            expected,
        },
    });
}

/// Compare inventory expected classes against a [`TensorClassifier`].
///
/// Manifest glob matching is the default classifier; a model plugin can
/// supply its own type without forking this function.
pub fn check_alignment_with<I, C>(inventory: &I, classifier: &C) -> AlignmentReport
where
    I: ModelInventory,
    C: TensorClassifier,
{
    let mut matched = 0;
    let mut expected_counts = ClassCounts::default();
    let mut actual_counts = ClassCounts::default();
    let mut mismatches = Vec::new();
    let mut boundary_summary: BTreeMap<String, usize> = BTreeMap::new();

    for t in inventory.tensors() {
        let actual = classifier.classify_name(&t.structural_name);
        expected_counts.bump(&t.expected_class);
        actual_counts.bump(&actual);
        if std::mem::discriminant(&t.expected_class) == std::mem::discriminant(&actual) {
            matched += 1;
        } else {
            record_mismatch(
                &mut mismatches,
                &mut boundary_summary,
                t.structural_name.clone(),
                t.expected_class.clone(),
                actual,
            );
        }
    }

    AlignmentReport {
        total_inventory_tensors: inventory.total_tensors(),
        matched,
        mismatched: mismatches.len(),
        preserve_expected_count: expected_counts.preserve,
        fp16_expected_count: expected_counts.fp16,
        ternary_expected_count: expected_counts.ternary,
        default_expected_count: expected_counts.default,
        preserve_actual_count: actual_counts.preserve,
        fp16_actual_count: actual_counts.fp16,
        ternary_actual_count: actual_counts.ternary,
        default_actual_count: actual_counts.default,
        mismatches,
        boundary_summary,
    }
}

/// Compare inventory expected classes against a dissect manifest.
///
/// `config.router_patterns` are unused while a manifest is supplied
/// (manifest globs win). The argument is kept so existing call sites stay
/// stable; plugins that need a non-manifest classifier should call
/// [`check_alignment_with`].
pub fn check_alignment<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    _config: &QuantizationConfig,
) -> AlignmentReport {
    check_alignment_with(inventory, manifest)
}

pub struct ConcreteCoverage {
    pub by_class: BTreeMap<String, usize>,
    pub total_classified: usize,
    pub unclassified: Vec<String>,
}

pub fn classify_full_inventory<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
    config: &QuantizationConfig,
) -> ConcreteCoverage {
    let mut by_class: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_classified = 0;
    let mut unclassified = Vec::new();

    for t in inventory.tensors() {
        let class = classify(&t.structural_name, Some(manifest), &config.router_patterns);
        let label = match &class {
            TensorClass::Preserve { .. } => "preserve",
            TensorClass::Fp16 { .. } => "fp16",
            TensorClass::TernaryCandidate { .. } => "ternary",
            TensorClass::Default => "default",
        };
        if let Some(count) = by_class.get_mut(label) {
            *count += 1;
        } else {
            by_class.insert(label.to_string(), 1);
        }
        total_classified += 1;
        if matches!(class, TensorClass::Default) {
            unclassified.push(t.structural_name.clone());
        }
    }

    ConcreteCoverage {
        by_class,
        total_classified,
        unclassified,
    }
}

#[cfg(test)]
pub(crate) use crate::core::test_support::plan_structural_manifest;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::grok1_inventory::Grok1Inventory;
    use crate::core::inventory::{InventoryTensor, VecInventory};
    use crate::types::GROK1_TENSOR_TOTAL;

    fn class_tensor(name: &str, expected: TensorClass) -> InventoryTensor {
        InventoryTensor {
            structural_name: name.into(),
            expected_class: expected,
            dtype: "f32",
            block: None,
            slot: None,
            kind: "fixture",
        }
    }

    /// Returns Fp16/Ternary/Default/Preserve for preserve/fp16/ternary/default
    /// so every [`ClassCounts::bump`] arm and [`record_mismatch`] run.
    struct CycleClassifier;

    impl TensorClassifier for CycleClassifier {
        fn classify_name(&self, name: &str) -> TensorClass {
            match name {
                "preserve" => TensorClass::Fp16 { reason: None },
                "fp16" => TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
                "ternary" => TensorClass::Default,
                _ => TensorClass::Preserve { reason: None },
            }
        }
    }

    #[test]
    fn check_alignment_with_records_class_mismatches() {
        let inv = VecInventory::from(vec![
            class_tensor("preserve", TensorClass::Preserve { reason: None }),
            class_tensor("fp16", TensorClass::Fp16 { reason: None }),
            class_tensor(
                "ternary",
                TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
            ),
            class_tensor("default", TensorClass::Default),
        ]);
        let report = check_alignment_with(&inv, &CycleClassifier);
        assert!(!report.is_aligned());
        assert_eq!((report.matched, report.mismatched), (0, 4));
        assert_eq!(
            (
                report.preserve_expected_count,
                report.fp16_expected_count,
                report.ternary_expected_count,
                report.default_expected_count
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(
            (
                report.preserve_actual_count,
                report.fp16_actual_count,
                report.ternary_actual_count,
                report.default_actual_count
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(report.boundary_summary.len(), 4);
        assert!(report.summary().contains("MISALIGNED"));
        assert!(matches!(
            report.mismatches[0].match_status,
            ClassMatch::Mismatch { .. }
        ));
    }

    #[test]
    fn full_inventory_has_770_tensors() {
        let inv = Grok1Inventory::full();
        assert_eq!(inv.len(), GROK1_TENSOR_TOTAL);
    }

    #[test]
    fn alignment_against_structural_manifest() {
        let report =
            crate::core::test_support::align_profile(&crate::core::models::grok1::Grok1Profile);

        eprintln!("{}", report.summary());
        for (boundary, count) in &report.boundary_summary {
            eprintln!("  boundary: {boundary} (x{count})");
        }

        assert_eq!(report.total_inventory_tensors, GROK1_TENSOR_TOTAL);
        assert!(
            report.is_aligned(),
            "all 770 tensors should match their expected class, got {} mismatches",
            report.mismatched
        );
    }

    #[test]
    fn preserve_tensors_are_not_ternary() {
        let report =
            crate::core::test_support::align_profile(&crate::core::models::grok1::Grok1Profile);

        let preserve_leaked_to_ternary = report
            .mismatches
            .iter()
            .filter(|m| {
                matches!(&m.expected_class, TensorClass::Preserve { .. })
                    && matches!(&m.actual_class, TensorClass::TernaryCandidate { .. })
            })
            .count();
        assert_eq!(
            preserve_leaked_to_ternary, 0,
            "preserve tensors must never be classified as ternary candidates"
        );
    }

    #[test]
    fn no_double_counting_in_inventory() {
        let inv = Grok1Inventory::full();
        let mut names = std::collections::BTreeSet::new();
        for t in &inv.tensors {
            assert!(
                names.insert(&t.structural_name),
                "duplicate tensor name: {}",
                t.structural_name
            );
        }
    }

    const EXPECTED_PRESERVE: usize = 321;
    const EXPECTED_FP16: usize = 0;
    const EXPECTED_TERNARY: usize = 449;
    const EXPECTED_DEFAULT: usize = 0;

    #[test]
    fn concrete_coverage_has_no_unclassified_tensors() {
        let inv = Grok1Inventory::full();
        let coverage = classify_full_inventory(
            &inv,
            embedded_grok1_structural_manifest(),
            &QuantizationConfig::default(),
        );

        assert_eq!(
            coverage.total_classified,
            EXPECTED_PRESERVE + EXPECTED_FP16 + EXPECTED_TERNARY + EXPECTED_DEFAULT
        );
        assert!(
            coverage.unclassified.is_empty(),
            "no tensors should fall to default; unclassified: {:?}",
            coverage.unclassified
        );
        assert_eq!(
            *coverage.by_class.get("preserve").unwrap_or(&0),
            EXPECTED_PRESERVE
        );
        assert_eq!(*coverage.by_class.get("fp16").unwrap_or(&0), EXPECTED_FP16);
        assert_eq!(
            *coverage.by_class.get("ternary").unwrap_or(&0),
            EXPECTED_TERNARY
        );
        assert_eq!(
            *coverage.by_class.get("default").unwrap_or(&0),
            EXPECTED_DEFAULT
        );
    }

    #[test]
    fn structural_manifest_loads() {
        let m = embedded_grok1_structural_manifest();
        assert_eq!(m.model.family, "grok-1");
        assert_eq!(m.preserve.len(), 6);
        assert_eq!(m.fp16.len(), 0);
        assert_eq!(m.ternary_candidates.len(), 8);
    }

    #[test]
    fn dry_run_structural_manifest_coverage_is_full() {
        use crate::core::dry_run::CoverageStatus;

        let report = plan_structural_manifest();

        assert_eq!(
            report.coverage.inventory_coverage,
            CoverageStatus::Full,
            "structural manifest dry-run should produce CoverageStatus::Full"
        );
        assert_eq!(report.coverage.covered_by_rules, GROK1_TENSOR_TOTAL);
    }

    #[test]
    fn dry_run_no_tensor_double_counted() {
        use crate::core::selection::glob_match;

        let report = plan_structural_manifest();
        let inv = Grok1Inventory::full();

        for t in &inv.tensors {
            let matching_rules: Vec<&str> = report
                .rule_plans
                .iter()
                .filter(|p| glob_match(&p.matcher, &t.structural_name))
                .map(|p| p.matcher.as_str())
                .collect();
            assert!(
                matching_rules.len() <= 1,
                "tensor '{}' matched by {} rules: {:?}",
                t.structural_name,
                matching_rules.len(),
                matching_rules
            );
        }
    }

    #[test]
    fn dry_run_router_tensors_are_preserve() {
        use crate::core::selection::glob_match;

        let report = plan_structural_manifest();
        let inv = Grok1Inventory::full();

        for t in &inv.tensors {
            if t.kind == "router" {
                let mut matched = false;
                for plan in &report.rule_plans {
                    if glob_match(&plan.matcher, &t.structural_name) {
                        matched = true;
                        assert!(
                            matches!(plan.class, TensorClass::Preserve { .. }),
                            "router tensor '{}' should be Preserve, got {:?}",
                            t.structural_name,
                            plan.class
                        );
                    }
                }
                assert!(
                    matched,
                    "router tensor '{}' was not matched by any manifest rule",
                    t.structural_name
                );
            }
        }
    }

    #[test]
    fn dry_run_expert_tensors_are_ternary() {
        use crate::core::selection::glob_match;

        let report = plan_structural_manifest();
        let inv = Grok1Inventory::full();

        for t in &inv.tensors {
            if t.kind.starts_with("moe_expert") || t.kind.starts_with("attn_proj_i8") {
                let mut matched = false;
                for plan in &report.rule_plans {
                    if glob_match(&plan.matcher, &t.structural_name) {
                        matched = true;
                        assert!(
                            matches!(plan.class, TensorClass::TernaryCandidate { .. }),
                            "expert tensor '{}' should be TernaryCandidate, got {:?}",
                            t.structural_name,
                            plan.class
                        );
                    }
                }
                assert!(
                    matched,
                    "expert tensor '{}' was not matched by any manifest rule",
                    t.structural_name
                );
            }
        }
    }

    #[test]
    fn dry_run_overcomplete_never_triggered_by_valid_manifest() {
        use crate::core::dry_run::CoverageStatus;

        let report = plan_structural_manifest();

        assert!(
            !matches!(
                report.coverage.inventory_coverage,
                CoverageStatus::OverComplete { .. }
            ),
            "valid structural manifest should never produce OverComplete"
        );
    }

    #[test]
    fn every_manifest_rule_matches_at_least_one_inventory_tensor() {
        use crate::core::selection::glob_match;

        let report = plan_structural_manifest();
        let inv = Grok1Inventory::full();

        for plan in &report.rule_plans {
            if plan.matcher == "<defaults>" {
                continue;
            }
            let matched = inv
                .tensors
                .iter()
                .any(|t| glob_match(&plan.matcher, &t.structural_name));
            assert!(
                matched,
                "manifest rule '{}' matches no inventory tensors (orphan rule)",
                plan.matcher
            );
        }
    }

    #[test]
    fn every_inventory_tensor_matched_by_manifest_rule() {
        use crate::core::selection::glob_match;

        let report = plan_structural_manifest();
        let inv = Grok1Inventory::full();

        let has_defaults = report.rule_plans.iter().any(|p| p.matcher == "<defaults>");

        for t in &inv.tensors {
            let matched_explicit = report
                .rule_plans
                .iter()
                .filter(|p| p.matcher != "<defaults>")
                .any(|p| glob_match(&p.matcher, &t.structural_name));
            assert!(
                matched_explicit || has_defaults,
                "inventory tensor '{}' not matched by any manifest rule (no explicit match and no <defaults> fallback)",
                t.structural_name
            );
        }
    }

    #[test]
    fn preserve_fp16_ternary_boundaries_match_xai_dissect() {
        let inv = Grok1Inventory::full();
        let (preserve, fp16, ternary, default) = inv.count_by_expected_class();

        assert_eq!(
            preserve, EXPECTED_PRESERVE,
            "{EXPECTED_PRESERVE} preserve: 64 routers + 256 block_norms + 1 final_norm"
        );
        assert_eq!(
            fp16, EXPECTED_FP16,
            "no fp16 tensors in structural manifest"
        );
        assert_eq!(
            ternary, EXPECTED_TERNARY,
            "{EXPECTED_TERNARY} ternary: 192 MoE expert + 256 attn_proj_i8 + 1 token_embedding"
        );
        assert_eq!(
            default, EXPECTED_DEFAULT,
            "no tensors should fall to default"
        );
    }

    #[test]
    fn no_router_tensor_classified_as_ternary() {
        let inv = Grok1Inventory::full();
        let manifest = embedded_grok1_structural_manifest();
        let classifier = manifest;

        for t in &inv.tensors {
            if t.kind == "router" {
                let actual = classifier.classify_name(&t.structural_name);
                assert!(
                    !matches!(actual, TensorClass::TernaryCandidate { .. }),
                    "router tensor '{}' must not be TernaryCandidate, got {:?}",
                    t.structural_name,
                    actual
                );
            }
        }
    }
}
