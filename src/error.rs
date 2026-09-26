use thiserror::Error;

#[derive(Debug, Error)]
pub enum GrokOzempicError {
    #[error("safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("expert index {index} out of range (num_experts={num_experts})")]
    ExpertOutOfRange { index: usize, num_experts: usize },

    #[error("quantization error: {0}")]
    Quantization(String),

    #[error("GOZ1 pack write error: {0}")]
    PackWrite(String),

    #[error("manifest I/O error at {path}: {source}")]
    ManifestIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("manifest parse error at {path}: {source}")]
    ManifestParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("unsupported manifest schema_version: got {got}, expected {expected}")]
    ManifestSchemaVersion { got: u32, expected: u32 },

    #[error("manifest tensor_name_convention mismatch: got {got:?}, expected {expected:?}")]
    ManifestNameConventionMismatch { got: String, expected: String },

    #[error("unsupported manifest precision tier: {got:?}")]
    ManifestInvalidPrecision { got: String },

    #[error(
        "tensor {name:?} matches no explicit rule in a fail-closed manifest; refusing \
         defaults fallthrough so preserve tensors (routers/norms) cannot be silently \
         ternary-quantized. Use names that match the manifest convention, or supply a \
         V1 `blk.*` manifest if defaults fallthrough is intended"
    )]
    ManifestV2UnmatchedTensor { name: String },

    #[error(
        "no explicit default precision is configured; refusing to silently ternary-quantize \
         unclassified tensors. Set manifest defaults.precision to ternary_snn, fp16, or \
         preserve, or list the tensor under preserve / fp16 / ternary_candidates"
    )]
    MissingDefaultPrecision,

    #[error(
        "ternary rule {pattern:?} matches mixed or unknown inventory dtypes; \
         wrap-vs-quantize cannot be decided from dtype"
    )]
    MixedInventoryDtype { pattern: String },

    #[error("artifact validation error: {0}")]
    ArtifactValidation(String),

    #[error("backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("hybrid stage planning error: {0}")]
    StagePlan(String),
}

pub type Result<T> = std::result::Result<T, GrokOzempicError>;
