//! grok-ozempic — SNN-logic quantization for MoE models
//!
//! Turns a massive Mixture-of-Experts into a sparse, membrane-driven, ternary
//! spiking system. Grok-1 is the reference [`core::model::ModelProfile`]; the
//! planner, alignment, and `BackendKernel` seams are model-agnostic. The GOZ1
//! on-disk layout is unchanged.
//!
//! The batch pipeline writes a **GOZ1** packed checkpoint (see [`core::weight_pack`]);
//! weights are expected from JAX/NumPy (`.npy`) or safetensors — see the README.

pub mod artifact;
pub mod core;
pub mod error;
pub mod reports;
pub mod types;

pub use core::HybridModel;
pub use core::alignment::{
    AlignmentReport, check_alignment, check_alignment_with, embedded_grok1_structural_manifest,
};
pub use core::backend::{BackendKernel, LocalBackend, MyelinBackend};
pub use core::dry_run::{
    CoverageStatus, CoverageSummary, DryRunPlanner, DryRunReport, OperationKind, PlannedKernelCall,
};
pub use core::grok1_inventory::Grok1Inventory;
pub use core::inventory::{InventoryTensor, ModelInventory, VecInventory};
pub use core::model::ModelProfile;
pub use core::models::grok1::Grok1Profile;
pub use core::models::toy_moe::{ToyMoeInventory, ToyMoeProfile};
pub use core::stage_planner::{
    HybridStagePlanner, PrecisionTier, StagePlan, StagePlanningReport, StageRequest,
    StubStagePlanner,
};
pub use types::{
    GROK1_HIDDEN_DIM, HybridConfig, HybridOutput, QuantizationConfig, QuantizationInputFormat,
    TelemetrySnapshot, TensorPrecision, quantize_goz1_config, validate_gif_threshold,
};

// Re-export main types for convenience
pub use crate::core::manifest::{
    ACCEPTED_NAME_CONVENTIONS, DissectManifest, Fp16Entry, GROK1_BASELINE_JSON,
    MANIFEST_NAME_CONVENTION_HF_MOE, MANIFEST_NAME_CONVENTION_V1, MANIFEST_NAME_CONVENTION_V2,
    MANIFEST_SCHEMA_VERSION, ManifestBlock, ManifestDefaults, ManifestModel, ManifestProducedBy,
    PreserveEntry, TernaryCandidate, embedded_grok1_baseline, is_accepted_name_convention,
    load_manifest, parse_manifest_bytes, uses_exact_inventory_counts,
};
pub use crate::core::ozempic::OzempicMoE;
pub use crate::core::precision::{
    decide as precision_decide, decide_for_tensor, parse_precision_str,
};
pub use crate::core::projector::Projector;
pub use crate::core::quantizer::{QuantizedTensor, quantize_f16, quantize_f32};
pub use crate::core::saaq::{SaaqTauEntry, SaaqTauMap, load_saaq_tau_map};
pub use crate::core::selection::{
    LEGACY_DEFAULT_ROUTER_PATTERNS, TensorClass, TensorClassifier, classify as selection_classify,
    glob_match,
};
pub use crate::core::stream::{ShardStats, append_grok1_arch_metadata, run_quantization};
pub use crate::core::weight_pack::{
    PackMetaValue, PackStreamWriter, PackTensorHeader, TENSOR_F16, TENSOR_TERNARY,
};
pub use crate::core::weight_pack_read::{PackVerifyReport, verify_pack_file};
