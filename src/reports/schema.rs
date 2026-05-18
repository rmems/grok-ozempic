use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub model_family: String,
    pub checkpoint: String,
    pub shards: usize,
    pub schema_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_experts: usize,
    pub d_ff: usize,
    pub n_blocks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorTotals {
    pub total: usize,
    pub f32_tensors: usize,
    pub int8_tensors: usize,
    pub quant_tensors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterEntry {
    pub block: usize,
    pub slot: usize,
    pub shape: (usize, usize),
    pub orientation: String,
    pub experts: usize,
    pub kind: String,
    pub structural_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertBlock {
    pub block: usize,
    pub experts: usize,
    pub expert_tensors: usize,
    pub slots: Vec<usize>,
    pub shapes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaaqTarget {
    pub rank: usize,
    pub tensor: String,
    pub kind: String,
    pub region: String,
    pub readiness: f64,
    pub opportunity: f64,
    pub risk: f64,
    pub disposition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaaqCritical {
    pub tensor: String,
    pub readiness: f64,
    pub risk: f64,
    pub reasons: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsEntry {
    pub tensor: String,
    pub kind: String,
    pub block: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactIR {
    pub manifest: ArtifactManifest,
    pub hyperparameters: Hyperparameters,
    pub totals: TensorTotals,
    pub routers: Vec<RouterEntry>,
    pub expert_blocks: Vec<ExpertBlock>,
    pub saaq_targets: Vec<SaaqTarget>,
    pub saaq_critical: Vec<SaaqCritical>,
    pub stats: Vec<StatsEntry>,
    pub mean_rms: f64,
}
