//! Backend-agnostic, weight-free planning contracts for hybrid MoE stages.
//!
//! These types extract the orchestration seam used by grok-ozempic's precision
//! policy. They describe intent only: no checkpoint is opened and no
//! quantization or GOZ1 packing is performed.

use serde::{Deserialize, Serialize};

use crate::{
    core::{manifest::DissectManifest, precision::decide_for_tensor, selection::TensorClass},
    error::{GrokOzempicError, Result},
    types::{QuantizationConfig, TensorPrecision},
};

/// Precision intent for a planned MoE tensor operation.
///
/// This is planning metadata derived from `src/core/precision.rs`; it contains
/// no packing or quantization behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionTier {
    /// Keep routing-critical data untouched by lossy transforms.
    #[default]
    Preserve,
    /// Retain the tensor in half precision.
    Fp16,
    /// Plan ternary SNN conversion in a concrete backend.
    TernarySnn,
}

impl PrecisionTier {
    /// Map the pipeline's resolved [`TensorPrecision`] onto planning metadata.
    pub const fn from_tensor_precision(precision: TensorPrecision) -> Self {
        match precision {
            TensorPrecision::Preserve => Self::Preserve,
            TensorPrecision::Fp16 => Self::Fp16,
            TensorPrecision::TernarySnn => Self::TernarySnn,
        }
    }
}

/// Description of one pipeline stage supplied to a [`HybridStagePlanner`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageRequest {
    /// Stable, human-readable pipeline stage name.
    pub name: String,
    /// Tensor classification already selected by the caller.
    pub tensor_class: TensorClass,
}

/// One dry-run operation selected for a pipeline stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StagePlan {
    /// Stage name copied from the request.
    pub stage_name: String,
    /// Precision a concrete backend should eventually use.
    pub precision: PrecisionTier,
    /// Effective GIF threshold from [`decide_for_tensor`], when the plan is
    /// ternary. `None` for preserve / fp16 so a backend cannot apply τ by
    /// accident.
    pub gif_threshold: Option<f32>,
    /// Always `false` for a conforming dry-run planner.
    pub weights_loaded: bool,
}

/// Weight-free aggregate returned by [`HybridStagePlanner::plan`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StagePlanningReport {
    /// Planned operations, in request order.
    pub stages: Vec<StagePlan>,
}

/// Maps classified pipeline stages to backend-neutral precision operations.
///
/// # Contract
///
/// Implementations must be deterministic for the same requests and must not
/// load weights, access checkpoint files, perform I/O, or quantize tensor data.
///
/// `TensorClass::Default` is resolved through [`decide_for_tensor`] so a
/// manifest's `defaults.precision` (fp16 / preserve / ternary_snn) is honored.
/// Missing default precision is [`GrokOzempicError::MissingDefaultPrecision`].
///
/// # Errors
///
/// Returns [`GrokOzempicError::StagePlan`] when a request cannot be mapped, or
/// forwards precision-policy errors from [`decide_for_tensor`].
pub trait HybridStagePlanner {
    /// Produce planning metadata for `stages` without executing the operations.
    fn plan(
        &self,
        stages: &[StageRequest],
        manifest: Option<&DissectManifest>,
        config: &QuantizationConfig,
    ) -> Result<StagePlanningReport>;
}

/// Deterministic reference planner: no weights, no matmul — pure planning metadata.
#[derive(Clone, Debug, Default)]
pub struct StubStagePlanner;

impl StubStagePlanner {
    /// Construct the weight-free reference planner.
    pub fn new() -> Self {
        Self
    }
}

fn plan_stage(
    stage: &StageRequest,
    manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> Result<StagePlan> {
    if stage.name.trim().is_empty() {
        return Err(GrokOzempicError::StagePlan(
            "stage name must not be empty".into(),
        ));
    }
    let (precision, gif_threshold) =
        decide_for_tensor(&stage.name, &stage.tensor_class, manifest, config)?;
    let precision = PrecisionTier::from_tensor_precision(precision);
    Ok(StagePlan {
        stage_name: stage.name.clone(),
        precision,
        gif_threshold: match precision {
            PrecisionTier::TernarySnn => Some(gif_threshold),
            PrecisionTier::Preserve | PrecisionTier::Fp16 => None,
        },
        weights_loaded: false,
    })
}

impl HybridStagePlanner for StubStagePlanner {
    fn plan(
        &self,
        stages: &[StageRequest],
        manifest: Option<&DissectManifest>,
        config: &QuantizationConfig,
    ) -> Result<StagePlanningReport> {
        let stages = stages
            .iter()
            .map(|stage| plan_stage(stage, manifest, config))
            .collect::<Result<Vec<_>>>()?;
        Ok(StagePlanningReport { stages })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::manifest::{
        DissectManifest, MANIFEST_NAME_CONVENTION_V1, MANIFEST_SCHEMA_VERSION, ManifestDefaults,
        ManifestModel,
    };

    fn manifest_with_defaults(precision: Option<&str>, gif: Option<f32>) -> DissectManifest {
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

    fn config() -> QuantizationConfig {
        QuantizationConfig::default()
    }

    fn plan(
        stages: &[StageRequest],
        manifest: Option<&DissectManifest>,
    ) -> Result<StagePlanningReport> {
        StubStagePlanner::new().plan(stages, manifest, &config())
    }

    #[test]
    fn stub_maps_all_precision_tiers_without_loading_weights() {
        let requests = [
            StageRequest {
                name: "router".into(),
                tensor_class: TensorClass::Preserve { reason: None },
            },
            StageRequest {
                name: "gate".into(),
                tensor_class: TensorClass::Fp16 { reason: None },
            },
            StageRequest {
                name: "expert".into(),
                tensor_class: TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
            },
        ];
        let report = plan(&requests, None).unwrap();
        assert_eq!(report.stages.len(), 3);
        assert_eq!(report.stages[0].precision, PrecisionTier::Preserve);
        assert_eq!(report.stages[0].gif_threshold, None);
        assert_eq!(report.stages[1].precision, PrecisionTier::Fp16);
        assert_eq!(report.stages[1].gif_threshold, None);
        assert_eq!(report.stages[2].precision, PrecisionTier::TernarySnn);
        assert_eq!(report.stages[2].gif_threshold, Some(config().gif_threshold));
        assert!(report.stages.iter().all(|stage| !stage.weights_loaded));
    }

    #[test]
    fn stub_is_deterministic_and_accepts_an_empty_inventory() {
        let planner = StubStagePlanner::new();
        let cfg = config();
        assert_eq!(
            planner.plan(&[], None, &cfg).unwrap(),
            StagePlanningReport::default()
        );
        let request = [StageRequest {
            name: "expert".into(),
            tensor_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
        }];
        assert_eq!(
            planner.plan(&request, None, &cfg).unwrap(),
            planner.plan(&request, None, &cfg).unwrap()
        );
    }

    #[test]
    fn stub_rejects_an_empty_stage_name_with_typed_error() {
        let error = plan(
            &[StageRequest {
                name: " ".into(),
                tensor_class: TensorClass::Default,
            }],
            None,
        )
        .unwrap_err();
        assert!(matches!(error, GrokOzempicError::StagePlan(_)));
    }

    #[test]
    fn stub_honors_manifest_default_precision_for_unclassified_tensors() {
        let request = [StageRequest {
            name: "unlisted".into(),
            tensor_class: TensorClass::Default,
        }];

        let fp16 = manifest_with_defaults(Some("fp16"), None);
        let fp16_report = plan(&request, Some(&fp16)).unwrap();
        assert_eq!(fp16_report.stages[0].precision, PrecisionTier::Fp16);
        assert_eq!(fp16_report.stages[0].gif_threshold, None);

        let preserve = manifest_with_defaults(Some("preserve"), None);
        let preserve_report = plan(&request, Some(&preserve)).unwrap();
        assert_eq!(preserve_report.stages[0].precision, PrecisionTier::Preserve);

        let ternary = manifest_with_defaults(Some("ternary_snn"), Some(0.08));
        let ternary_report = plan(&request, Some(&ternary)).unwrap();
        assert_eq!(
            ternary_report.stages[0].precision,
            PrecisionTier::TernarySnn
        );
        assert_eq!(ternary_report.stages[0].gif_threshold, Some(0.08));
    }

    #[test]
    fn stub_refuses_default_class_without_manifest_precision() {
        let error = plan(
            &[StageRequest {
                name: "unlisted".into(),
                tensor_class: TensorClass::Default,
            }],
            None,
        )
        .unwrap_err();
        assert!(
            matches!(error, GrokOzempicError::MissingDefaultPrecision),
            "got {error:?}"
        );
    }

    #[test]
    fn stub_carries_ternary_candidate_gif_threshold() {
        let report = plan(
            &[StageRequest {
                name: "expert".into(),
                tensor_class: TensorClass::TernaryCandidate {
                    rank: Some(0.98),
                    gif_threshold: Some(0.04),
                },
            }],
            None,
        )
        .unwrap();
        assert_eq!(report.stages[0].precision, PrecisionTier::TernarySnn);
        assert_eq!(report.stages[0].gif_threshold, Some(0.04));
    }
}
