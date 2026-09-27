//! Shared test harness for planner / alignment tests.
//!
//! Model-specific fixtures live in `crate::core::models`. This module only
//! provides generic helpers so adding a family does not copy-paste the
//! three-line `inventory + manifest + config` setup (GH #32 / RM-65).

use crate::core::alignment::{AlignmentReport, check_alignment};
use crate::core::dry_run::{DryRunPlanner, DryRunReport};
use crate::core::inventory::ModelInventory;
use crate::core::manifest::{
    DissectManifest, MANIFEST_NAME_CONVENTION_V1, MANIFEST_SCHEMA_VERSION, ManifestDefaults,
    ManifestModel,
};
use crate::core::model::ModelProfile;
use crate::types::QuantizationConfig;

/// Minimal manifest with only `defaults` populated (precision tests, stage planner).
pub(crate) fn manifest_with_defaults(precision: Option<&str>, gif: Option<f32>) -> DissectManifest {
    DissectManifest {
        schema: "xai-dissect.manifest".into(),
        schema_version: MANIFEST_SCHEMA_VERSION,
        model: ManifestModel {
            family: "grok-1".into(),
            source: "xai-org/grok-1".into(),
            tensor_name_convention: MANIFEST_NAME_CONVENTION_V1.into(),
        },
        produced_by: None,
        defaults: ManifestDefaults {
            precision: precision.map(String::from),
            gif_threshold: gif,
        },
        preserve: vec![],
        fp16: vec![],
        ternary_candidates: vec![],
        blocks: vec![],
    }
}

/// Dry-run a profile through the same planner Grok-1 uses.
pub(crate) fn plan_profile<P: ModelProfile>(profile: &P) -> DryRunReport {
    DryRunPlanner::plan(
        &profile.inventory(),
        &profile.manifest(),
        &profile.quantization_config(),
    )
    .expect("plan should succeed")
}

/// Align a profile's inventory against its own manifest.
pub(crate) fn align_profile<P: ModelProfile>(profile: &P) -> AlignmentReport {
    check_alignment(
        &profile.inventory(),
        &profile.manifest(),
        &profile.quantization_config(),
    )
}

/// Dry-run an arbitrary inventory + manifest with default config.
pub(crate) fn plan_for<I: ModelInventory>(
    inventory: &I,
    manifest: &DissectManifest,
) -> DryRunReport {
    DryRunPlanner::plan(inventory, manifest, &QuantizationConfig::default())
        .expect("plan should succeed")
}

/// Grok-1 structural dry-run used by existing `alignment` / `dry_run` tests.
pub(crate) fn plan_structural_manifest() -> DryRunReport {
    plan_profile(&crate::core::models::grok1::Grok1Profile)
}
