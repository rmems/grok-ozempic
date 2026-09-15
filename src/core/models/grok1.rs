//! Grok-1 [`crate::core::model::ModelProfile`] — the reference plugin.

use crate::core::alignment::embedded_grok1_structural_manifest;
use crate::core::grok1_inventory::Grok1Inventory;
use crate::core::manifest::{DissectManifest, MANIFEST_NAME_CONVENTION_V2};
use crate::core::model::ModelProfile;

/// Reference profile: 770-tensor Grok-1 structural V2 inventory + manifest.
#[derive(Debug, Clone, Copy, Default)]
pub struct Grok1Profile;

impl ModelProfile for Grok1Profile {
    type Inventory = Grok1Inventory;

    fn family(&self) -> &'static str {
        "grok-1"
    }

    fn source(&self) -> &'static str {
        "xai-org/grok-1"
    }

    fn tensor_name_convention(&self) -> &'static str {
        MANIFEST_NAME_CONVENTION_V2
    }

    fn inventory(&self) -> Grok1Inventory {
        Grok1Inventory::full()
    }

    fn manifest(&self) -> DissectManifest {
        embedded_grok1_structural_manifest().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dry_run::CoverageStatus;
    use crate::core::test_support::{align_profile, plan_profile, plan_structural_manifest};
    use crate::types::GROK1_TENSOR_TOTAL;

    #[test]
    fn profile_matches_structural_harness() {
        let profile = Grok1Profile;
        assert_eq!(profile.family(), "grok-1");
        assert_eq!(
            profile.tensor_name_convention(),
            MANIFEST_NAME_CONVENTION_V2
        );
        assert_eq!(profile.inventory().len(), GROK1_TENSOR_TOTAL);

        let via_profile = plan_profile(&profile);
        let via_helper = plan_structural_manifest();
        assert_eq!(
            via_profile.coverage.inventory_total,
            via_helper.coverage.inventory_total
        );
        assert_eq!(
            via_profile.coverage.inventory_coverage,
            CoverageStatus::Full
        );
        assert!(align_profile(&profile).is_aligned());
    }
}
