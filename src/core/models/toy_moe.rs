//! Minimal Mixtral-style MoE used as the second in-tree [`ModelProfile`].
//!
//! Two layers × two experts, HuggingFace `model.layers.{L}.{module}.{param}`
//! names. All source dtypes are `f32` so ternary rules plan as quantize
//! (not wrap). GIF default is `0.10` so the profile is not Grok-1's `0.05`.

use std::sync::OnceLock;

use crate::core::inventory::{InventoryTensor, VecInventory};
use crate::core::manifest::{
    DissectManifest, MANIFEST_NAME_CONVENTION_HF_MOE, parse_manifest_bytes,
};
use crate::core::model::ModelProfile;
use crate::core::selection::TensorClass;

/// HuggingFace-style family id used on the embedded toy manifest.
pub const TOY_MOE_FAMILY: &str = "toy-moe";

/// Layers in the toy inventory (not a production Mixtral clone).
pub const TOY_MOE_LAYERS: u32 = 2;
/// Experts per MoE block.
pub const TOY_MOE_EXPERTS: u32 = 2;
/// `embed` + `lm_head` + `final_norm` + `2 * (2 norms + 1 gate + 4 attn + 6 expert)`.
pub const TOY_MOE_TENSOR_TOTAL: usize = 29;
/// Preserve: 2 layers × (gate + 2 norms) + final norm.
pub const TOY_MOE_PRESERVE: usize = 7;
/// Ternary: embed + lm_head + 2 layers × (4 attn + 6 expert).
pub const TOY_MOE_TERNARY: usize = 22;
/// Distinct from Grok-1's 0.05 so per-profile GIF defaults are testable.
pub const TOY_MOE_GIF_THRESHOLD: f32 = 0.10;

const TOY_MOE_MANIFEST_JSON: &str = r#"{
    "schema": "xai-dissect.manifest",
    "schema_version": 1,
    "model": {
        "family": "toy-moe",
        "source": "grok-ozempic/toy-moe",
        "tensor_name_convention": "model.layers.{L}.{module}.{param}"
    },
    "defaults": {
        "precision": "ternary_snn",
        "gif_threshold": 0.10
    },
    "preserve": [
        { "name": "model.layers.*.block_sparse_moe.gate.weight", "reason": "moe-router" },
        { "name": "model.layers.*.input_layernorm.weight", "reason": "norm" },
        { "name": "model.layers.*.post_attention_layernorm.weight", "reason": "norm" },
        { "name": "model.norm.weight", "reason": "final-norm" }
    ],
    "fp16": [],
    "ternary_candidates": [
        { "name": "model.embed_tokens.weight" },
        { "name": "lm_head.weight" },
        { "name": "model.layers.*.self_attn.q_proj.weight" },
        { "name": "model.layers.*.self_attn.k_proj.weight" },
        { "name": "model.layers.*.self_attn.v_proj.weight" },
        { "name": "model.layers.*.self_attn.o_proj.weight" },
        { "name": "model.layers.*.block_sparse_moe.experts.*.w1.weight" },
        { "name": "model.layers.*.block_sparse_moe.experts.*.w2.weight" },
        { "name": "model.layers.*.block_sparse_moe.experts.*.w3.weight" }
    ]
}"#;

/// Second in-tree model: Mixtral-like names, not Grok-1 slots.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToyMoeProfile;

/// Owned inventory wrapper around the toy tensor list.
pub type ToyMoeInventory = VecInventory;

impl ToyMoeProfile {
    /// Embedded toy manifest, parsed through the same loader as Grok-1.
    pub fn embedded_manifest() -> &'static DissectManifest {
        static CACHE: OnceLock<DissectManifest> = OnceLock::new();
        CACHE.get_or_init(|| {
            parse_manifest_bytes(
                TOY_MOE_MANIFEST_JSON.as_bytes(),
                "<embedded toy-moe manifest>",
            )
            .expect("embedded toy-moe manifest must parse")
        })
    }
}

impl ModelProfile for ToyMoeProfile {
    type Inventory = VecInventory;

    fn family(&self) -> &'static str {
        TOY_MOE_FAMILY
    }

    fn source(&self) -> &'static str {
        "grok-ozempic/toy-moe"
    }

    fn tensor_name_convention(&self) -> &'static str {
        MANIFEST_NAME_CONVENTION_HF_MOE
    }

    fn inventory(&self) -> VecInventory {
        VecInventory::new(build_toy_moe_tensors())
    }

    fn manifest(&self) -> DissectManifest {
        Self::embedded_manifest().clone()
    }

    fn default_gif_threshold(&self) -> f32 {
        TOY_MOE_GIF_THRESHOLD
    }
}

fn tensor(
    name: String,
    class: TensorClass,
    dtype: &'static str,
    block: Option<u32>,
    slot: Option<u32>,
    kind: &'static str,
) -> InventoryTensor {
    InventoryTensor {
        structural_name: name,
        expected_class: class,
        dtype,
        block,
        slot,
        kind,
    }
}

fn preserve(reason: &str) -> TensorClass {
    TensorClass::Preserve {
        reason: Some(reason.to_string()),
    }
}

fn ternary() -> TensorClass {
    TensorClass::TernaryCandidate {
        rank: None,
        gif_threshold: None,
    }
}

fn push_layer(tensors: &mut Vec<InventoryTensor>, layer: u32) {
    tensors.push(tensor(
        format!("model.layers.{layer}.input_layernorm.weight"),
        preserve("norm"),
        "f32",
        Some(layer),
        None,
        "input_norm",
    ));
    for (proj, kind) in [
        ("q_proj", "attn_q"),
        ("k_proj", "attn_k"),
        ("v_proj", "attn_v"),
        ("o_proj", "attn_o"),
    ] {
        tensors.push(tensor(
            format!("model.layers.{layer}.self_attn.{proj}.weight"),
            ternary(),
            "f32",
            Some(layer),
            None,
            kind,
        ));
    }
    tensors.push(tensor(
        format!("model.layers.{layer}.post_attention_layernorm.weight"),
        preserve("norm"),
        "f32",
        Some(layer),
        None,
        "post_attn_norm",
    ));
    tensors.push(tensor(
        format!("model.layers.{layer}.block_sparse_moe.gate.weight"),
        preserve("moe-router"),
        "f32",
        Some(layer),
        None,
        "router",
    ));
    push_experts(tensors, layer);
}

fn push_experts(tensors: &mut Vec<InventoryTensor>, layer: u32) {
    for expert in 0..TOY_MOE_EXPERTS {
        for (proj, kind) in [
            ("w1", "moe_expert.w1"),
            ("w2", "moe_expert.w2"),
            ("w3", "moe_expert.w3"),
        ] {
            tensors.push(tensor(
                format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight"),
                ternary(),
                "f32",
                Some(layer),
                Some(expert),
                kind,
            ));
        }
    }
}

fn build_toy_moe_tensors() -> Vec<InventoryTensor> {
    let mut tensors = Vec::with_capacity(TOY_MOE_TENSOR_TOTAL);
    tensors.push(tensor(
        "model.embed_tokens.weight".into(),
        ternary(),
        "f32",
        None,
        None,
        "embed",
    ));
    for layer in 0..TOY_MOE_LAYERS {
        push_layer(&mut tensors, layer);
    }
    tensors.push(tensor(
        "model.norm.weight".into(),
        preserve("final-norm"),
        "f32",
        None,
        None,
        "final_norm",
    ));
    tensors.push(tensor(
        "lm_head.weight".into(),
        ternary(),
        "f32",
        None,
        None,
        "lm_head",
    ));
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dry_run::{CoverageStatus, OperationKind};
    use crate::core::inventory::ModelInventory;
    use crate::core::selection::glob_match;
    use crate::core::test_support::{align_profile, plan_profile};
    use crate::types::QuantizationConfig;

    #[test]
    fn inventory_counts_match_the_layout() {
        let inv = ToyMoeProfile.inventory();
        assert_eq!(inv.total_tensors(), TOY_MOE_TENSOR_TOTAL);
        let (preserve_n, fp16, ternary_n, default) = inv.count_by_expected_class();
        assert_eq!(preserve_n, TOY_MOE_PRESERVE);
        assert_eq!(fp16, 0);
        assert_eq!(ternary_n, TOY_MOE_TERNARY);
        assert_eq!(default, 0);
    }

    #[test]
    fn no_duplicate_names() {
        let inv = ToyMoeProfile.inventory();
        let mut seen = std::collections::BTreeSet::new();
        for t in inv.tensors() {
            assert!(
                seen.insert(&t.structural_name),
                "duplicate: {}",
                t.structural_name
            );
        }
    }

    #[test]
    fn embedded_manifest_uses_hf_moe_convention() {
        let m = ToyMoeProfile::embedded_manifest();
        assert_eq!(ToyMoeProfile.family(), TOY_MOE_FAMILY);
        assert_eq!(ToyMoeProfile.source(), "grok-ozempic/toy-moe");
        assert_eq!(
            ToyMoeProfile.tensor_name_convention(),
            MANIFEST_NAME_CONVENTION_HF_MOE
        );
        assert_eq!(m.model.family, TOY_MOE_FAMILY);
        assert_eq!(m.model.source, "grok-ozempic/toy-moe");
        assert_eq!(
            m.model.tensor_name_convention,
            MANIFEST_NAME_CONVENTION_HF_MOE
        );
        assert_eq!(m.defaults.gif_threshold, Some(TOY_MOE_GIF_THRESHOLD));
    }

    #[test]
    fn alignment_is_full() {
        let report = align_profile(&ToyMoeProfile);
        assert_eq!(report.total_inventory_tensors, TOY_MOE_TENSOR_TOTAL);
        assert!(
            report.is_aligned(),
            "toy-moe should align, got {} mismatches: {:?}",
            report.mismatched,
            report
                .mismatches
                .iter()
                .map(|m| m.structural_name.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn dry_run_covers_every_tensor() {
        let report = plan_profile(&ToyMoeProfile);
        assert_eq!(report.coverage.inventory_total, TOY_MOE_TENSOR_TOTAL);
        assert_eq!(report.coverage.inventory_coverage, CoverageStatus::Full);
        assert_eq!(report.coverage.covered_by_rules, TOY_MOE_TENSOR_TOTAL);
    }

    #[test]
    fn gate_rule_counts_exactly_two_layers() {
        let report = plan_profile(&ToyMoeProfile);
        let gate = report
            .rule_plans
            .iter()
            .find(|p| p.matcher.contains("gate.weight"))
            .expect("gate preserve rule");
        assert_eq!(gate.estimated_tensor_count, TOY_MOE_LAYERS as usize);
        assert!(matches!(gate.class, TensorClass::Preserve { .. }));
        assert_eq!(gate.operation, OperationKind::ConvertFp16);
    }

    #[test]
    fn f32_experts_quantize_not_wrap() {
        let report = plan_profile(&ToyMoeProfile);
        for plan in &report.rule_plans {
            if plan.matcher.contains("experts") {
                assert_eq!(
                    plan.operation,
                    OperationKind::QuantizeTernary,
                    "f32 expert rule '{}' should quantize, got {}",
                    plan.matcher,
                    plan.operation
                );
                assert_eq!(plan.gif_threshold, TOY_MOE_GIF_THRESHOLD);
            }
        }
    }

    #[test]
    fn routers_never_match_ternary_rules() {
        let report = plan_profile(&ToyMoeProfile);
        let inv = ToyMoeProfile.inventory();
        for t in inv.tensors() {
            if t.kind != "router" {
                continue;
            }
            let ternary_hit = report.rule_plans.iter().any(|p| {
                matches!(p.class, TensorClass::TernaryCandidate { .. })
                    && glob_match(&p.matcher, &t.structural_name)
            });
            assert!(
                !ternary_hit,
                "router {} matched a ternary rule",
                t.structural_name
            );
        }
    }

    #[test]
    fn profile_gif_differs_from_engine_default() {
        let cfg = ToyMoeProfile.quantization_config();
        assert_eq!(cfg.gif_threshold, TOY_MOE_GIF_THRESHOLD);
        assert!(
            (cfg.gif_threshold - QuantizationConfig::default().gif_threshold).abs() > f32::EPSILON
        );
        assert!(!cfg.use_embedded_baseline);
    }
}
