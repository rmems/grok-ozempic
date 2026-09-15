//! In-tree [`crate::core::model::ModelProfile`] plugins.
//!
//! Model-specific tests live in these modules so they do not duplicate
//! planner setup across `dry_run` / `alignment`. Grok-1 remains the
//! reference; [`toy_moe`] is the second family required by GH #32 / RM-65.

pub mod grok1;
pub mod toy_moe;

#[cfg(test)]
mod tests {
    use crate::core::dry_run::CoverageStatus;
    use crate::core::model::ModelProfile;
    use crate::core::models::grok1::Grok1Profile;
    use crate::core::models::toy_moe::{TOY_MOE_TENSOR_TOTAL, ToyMoeProfile};
    use crate::core::test_support::plan_profile;
    use crate::types::GROK1_TENSOR_TOTAL;

    #[test]
    fn two_profiles_plan_independently() {
        let grok = plan_profile(&Grok1Profile);
        let toy = plan_profile(&ToyMoeProfile);
        assert_eq!(grok.coverage.inventory_total, GROK1_TENSOR_TOTAL);
        assert_eq!(toy.coverage.inventory_total, TOY_MOE_TENSOR_TOTAL);
        assert_ne!(
            grok.coverage.inventory_total, toy.coverage.inventory_total,
            "second model must not reuse the Grok-1 770-tensor layout"
        );
        assert_eq!(grok.coverage.inventory_coverage, CoverageStatus::Full);
        assert_eq!(toy.coverage.inventory_coverage, CoverageStatus::Full);
        assert_ne!(Grok1Profile.family(), ToyMoeProfile.family());
        assert_ne!(Grok1Profile.source(), ToyMoeProfile.source());
        assert_ne!(
            Grok1Profile.tensor_name_convention(),
            ToyMoeProfile.tensor_name_convention()
        );
    }
}
