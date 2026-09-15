//! Model plugins for the quantization engine.
//!
//! [`ModelProfile`] is the seam a higher-level registry (for example
//! [magere-brug](https://github.com/rmems/magere-brug)) uses to add a model
//! without forking this crate. Grok-1 is the reference implementation;
//! [`crate::core::models::toy_moe`] is a second inventory that proves dry-run
//! planning and alignment are not Grok-shaped.
//!
//! The GOZ1 on-disk format and the Grok-1 packing CLI stay adapters. New
//! models plug in at this trait — they do not change `weight_pack`.

use crate::core::inventory::ModelInventory;
use crate::core::manifest::DissectManifest;
use crate::types::QuantizationConfig;

/// Per-model adapter: inventory, naming convention, and default quant config.
///
/// Implement this for a new MoE family, then drive [`crate::core::dry_run::DryRunPlanner`]
/// and [`crate::core::alignment::check_alignment`] with [`Self::inventory`] and
/// [`Self::manifest`]. Classification stays manifest-glob driven
/// ([`crate::core::selection::classify`]); override the globs, not the planner.
pub trait ModelProfile {
    /// Concrete tensor inventory for this family.
    type Inventory: ModelInventory;

    /// Manifest `model.family` (for example `"grok-1"` or `"toy-moe"`).
    fn family(&self) -> &'static str;

    /// Provenance string stored on the manifest (`model.source`).
    fn source(&self) -> &'static str;

    /// Value of `model.tensor_name_convention`. Must be listed in
    /// [`crate::core::manifest::ACCEPTED_NAME_CONVENTIONS`].
    fn tensor_name_convention(&self) -> &'static str;

    /// Full tensor inventory used by dry-run coverage and alignment.
    fn inventory(&self) -> Self::Inventory;

    /// Classification manifest (preserve / fp16 / ternary globs).
    fn manifest(&self) -> DissectManifest;

    /// Default GIF saliency ratio when the manifest omits `defaults.gif_threshold`.
    fn default_gif_threshold(&self) -> f32 {
        0.05
    }

    /// Engine config with this profile's GIF default. Does **not** set
    /// [`QuantizationConfig::use_embedded_baseline`] — that flag is the
    /// Grok-1 CLI adapter, not a generic engine knob.
    fn quantization_config(&self) -> QuantizationConfig {
        QuantizationConfig {
            gif_threshold: self.default_gif_threshold(),
            ..QuantizationConfig::default()
        }
    }
}
