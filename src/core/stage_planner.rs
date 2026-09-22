//! Backend-agnostic, weight-free planning contracts for hybrid MoE stages.
//!
//! These types extract the orchestration seam used by grok-ozempic's precision
//! policy. They describe intent only: no checkpoint is opened and no
//! quantization or GOZ1 packing is performed.

use serde::{Deserialize, Serialize};

use crate::{
    core::selection::TensorClass,
    error::{GrokOzempicError, Result},
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

/// Description of one pipeline stage supplied to a [`HybridStagePlanner`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageRequest {
    /// Stable, human-readable pipeline stage name.
    pub name: String,
    /// Tensor classification already selected by the caller.
    pub tensor_class: TensorClass,
}

/// One dry-run operation selected for a pipeline stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagePlan {
    /// Stage name copied from the request.
    pub stage_name: String,
    /// Precision a concrete backend should eventually use.
    pub precision: PrecisionTier,
    /// Always `false` for a conforming dry-run planner.
    pub weights_loaded: bool,
}

/// Weight-free aggregate returned by [`HybridStagePlanner::plan`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
/// # Errors
///
/// Returns [`GrokOzempicError::StagePlan`] when a request cannot be mapped.
pub trait HybridStagePlanner {
    /// Produce planning metadata for `stages` without executing the operations.
    fn plan(&self, stages: &[StageRequest]) -> Result<StagePlanningReport>;
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

impl HybridStagePlanner for StubStagePlanner {
    fn plan(&self, stages: &[StageRequest]) -> Result<StagePlanningReport> {
        let stages = stages
            .iter()
            .map(|stage| {
                if stage.name.trim().is_empty() {
                    return Err(GrokOzempicError::StagePlan(
                        "stage name must not be empty".into(),
                    ));
                }
                let precision = match &stage.tensor_class {
                    TensorClass::Preserve { .. } => PrecisionTier::Preserve,
                    TensorClass::Fp16 { .. } => PrecisionTier::Fp16,
                    TensorClass::TernaryCandidate { .. } | TensorClass::Default => {
                        PrecisionTier::TernarySnn
                    }
                };
                Ok(StagePlan {
                    stage_name: stage.name.clone(),
                    precision,
                    weights_loaded: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(StagePlanningReport { stages })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let report = StubStagePlanner::new().plan(&requests).unwrap();
        assert_eq!(report.stages.len(), 3);
        assert_eq!(report.stages[0].precision, PrecisionTier::Preserve);
        assert_eq!(report.stages[1].precision, PrecisionTier::Fp16);
        assert_eq!(report.stages[2].precision, PrecisionTier::TernarySnn);
        assert!(report.stages.iter().all(|stage| !stage.weights_loaded));
    }

    #[test]
    fn stub_is_deterministic_and_accepts_an_empty_inventory() {
        let planner = StubStagePlanner::new();
        assert_eq!(planner.plan(&[]).unwrap(), StagePlanningReport::default());
        let request = [StageRequest {
            name: "default".into(),
            tensor_class: TensorClass::Default,
        }];
        assert_eq!(
            planner.plan(&request).unwrap(),
            planner.plan(&request).unwrap()
        );
    }

    #[test]
    fn stub_rejects_an_empty_stage_name_with_typed_error() {
        let error = StubStagePlanner::new()
            .plan(&[StageRequest {
                name: " ".into(),
                tensor_class: TensorClass::Default,
            }])
            .unwrap_err();
        assert!(matches!(error, GrokOzempicError::StagePlan(_)));
    }
}
