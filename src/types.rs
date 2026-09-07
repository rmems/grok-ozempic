use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Quantization pipeline types
// ---------------------------------------------------------------------------

/// Controls which precision is applied to a given tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorPrecision {
    /// Two-bit ternary {-1, 0, +1} with saliency-gated GIF threshold.
    TernarySnN,
    /// Keep original FP16 — used for MoE routing gates.
    Fp16,
    /// Routing-critical / no-touch tier. Populated when the
    /// `xai-dissect` manifest's `preserve` list matches a tensor
    /// (typically MoE routers, expert gates, attention readouts).
    ///
    /// **GOZ1 v1 on-disk encoding:** identical FP16 bytes to
    /// [`TensorPrecision::Fp16`] — both tiers serialize through the
    /// same FP16 writer path and use `TENSOR_F16` in the GOZ1 tensor
    /// table. This is the **final, documented behavior** for GOZ1 v1,
    /// not a transitional shortcut.
    ///
    /// The variant is kept distinct from [`TensorPrecision::Fp16`] for
    /// three reasons:
    /// 1. **Manifest-intent traceability** — pipeline diagnostics can
    ///    tell which manifest list claimed a tensor.
    /// 2. **Policy guarantee** — `Preserve` signals "must never be
    ///    ternary-quantized" even if a future policy change shifts the
    ///    default for `Fp16`.
    /// 3. **Forward compatibility** — a future GOZ1 format version may
    ///    promote `Preserve` to true source-dtype passthrough
    ///    (F32/BF16 kept as-is) without an API rename or migration.
    Preserve,
}

/// Weight container layout for [`QuantizationConfig::input_dir`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationInputFormat {
    /// Hugging Face–style `*.safetensors` shards (memory-mapped).
    #[default]
    Safetensors,
    /// Directory of per-tensor `*.npy` files — **primary layout for JAX/Flax** (NumPy export).
    /// Use `__` in the filename stem in place of `.` in tensor names
    /// (e.g. `blk__0__weight.npy` → `blk.0.weight`).
    NpyDir,
}

/// Configuration for the out-of-core quantization pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantizationConfig {
    /// Directory that holds weight shards (see [`QuantizationInputFormat`]).
    pub input_dir: String,
    /// Path for the output **GOZ1** packed checkpoint (see `weight_pack`).
    pub output_path: String,
    /// GIF saliency threshold ratio: weights with |w| < threshold × rms(layer)
    /// are silenced to 0; the rest become ±1.
    pub gif_threshold: f32,
    /// Tensor name substrings that identify routing / gate tensors which should
    /// remain in FP16 instead of being ternary-quantized.
    ///
    /// **Legacy field.** When an `xai-dissect` manifest is supplied via
    /// [`QuantizationConfig::manifest_path`], the manifest wins and this
    /// list is ignored. A deprecation log line will be emitted by the
    /// selection seam introduced in phase 2.
    pub router_patterns: Vec<String>,
    /// Input layout: safetensors shards vs flat `.npy` tensors.
    pub input_format: QuantizationInputFormat,
    /// Optional path to an `xai-dissect` JSON manifest (schema v1).
    ///
    /// **Reserved in phase 1.** The field is exposed so callers can start
    /// plumbing manifests through configuration, but
    /// [`crate::core::stream::run_quantization`] does **not** consume it
    /// yet. Wiring lands in phase 2 via dedicated selection and precision
    /// modules.
    ///
    /// Precedence (phase 2+): this explicit path > `GROK_OZEMPIC_MANIFEST`
    /// env var > in-tree `dissect/grok-1/baseline.json` fallback (only
    /// when [`Self::use_embedded_baseline`] is `true`) > legacy
    /// `router_patterns` heuristic.
    #[serde(default)]
    pub manifest_path: Option<PathBuf>,
    /// Opt in to the compiled-in non-authoritative Grok-1 baseline
    /// manifest as a fallback when neither
    /// [`Self::manifest_path`] nor `GROK_OZEMPIC_MANIFEST` is set.
    ///
    /// Default is `false` so upgrading from phase 1 preserves existing
    /// legacy-heuristic behavior. Set to `true` for a Grok-1 export to
    /// pick up the reference manifest without pointing at a file.
    #[serde(default)]
    pub use_embedded_baseline: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            input_dir: String::new(),
            output_path: String::new(),
            gif_threshold: 0.05,
            router_patterns: Vec::new(),
            input_format: QuantizationInputFormat::Safetensors,
            manifest_path: None,
            use_embedded_baseline: false,
        }
    }
}

/// Build a [`QuantizationConfig`] for the GOZ1 CLI (`quantize-goz1`) and tests.
///
/// Used by the `cli` binary and unit-tested so flag mapping stays stable for the
/// first embedding experiment (GitHub #38 / Linear RM-193).
pub fn quantize_goz1_config(
    input_dir: impl Into<String>,
    output_path: impl Into<String>,
    input_format: QuantizationInputFormat,
    manifest_path: Option<PathBuf>,
    gif_threshold: Option<f32>,
    use_embedded_baseline: bool,
) -> QuantizationConfig {
    let mut cfg = QuantizationConfig {
        input_dir: input_dir.into(),
        output_path: output_path.into(),
        input_format,
        manifest_path,
        use_embedded_baseline,
        ..QuantizationConfig::default()
    };
    if let Some(t) = gif_threshold {
        cfg.gif_threshold = t;
    }
    cfg
}

/// Return an error message if `gif_threshold` is unusable for ternary GIF gating.
pub fn validate_gif_threshold(t: f32) -> Result<(), String> {
    if !t.is_finite() {
        return Err(format!("gif_threshold must be finite (got {t})"));
    }
    if t < 0.0 {
        return Err(format!("gif_threshold must be >= 0 (got {t})"));
    }
    Ok(())
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    pub timestamp_ms: u64,
}

impl TelemetrySnapshot {
    pub fn thermal_stress(&self) -> f32 {
        ((self.gpu_temp_c - 60.0) / 30.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod quantize_goz1_config_tests {
    use super::*;

    #[test]
    fn quantize_goz1_config_maps_cli_fields() {
        let cfg = quantize_goz1_config(
            "/tmp/in",
            "/tmp/out.goz1",
            QuantizationInputFormat::NpyDir,
            Some(PathBuf::from("dissect/grok-1/baseline.json")),
            Some(0.1),
            true,
        );
        assert_eq!(cfg.input_dir, "/tmp/in");
        assert_eq!(cfg.output_path, "/tmp/out.goz1");
        assert_eq!(cfg.input_format, QuantizationInputFormat::NpyDir);
        assert_eq!(
            cfg.manifest_path,
            Some(PathBuf::from("dissect/grok-1/baseline.json"))
        );
        assert!((cfg.gif_threshold - 0.1).abs() < f32::EPSILON);
        assert!(cfg.use_embedded_baseline);
    }

    #[test]
    fn quantize_goz1_config_omitted_gif_keeps_default() {
        let cfg = quantize_goz1_config(
            "/tmp/in",
            "/tmp/out.goz1",
            QuantizationInputFormat::NpyDir,
            None,
            None,
            false,
        );
        assert!((cfg.gif_threshold - 0.05).abs() < f32::EPSILON);
        assert!(!cfg.use_embedded_baseline);
        assert!(cfg.manifest_path.is_none());
    }

    #[test]
    fn validate_gif_threshold_rejects_nan_and_negative() {
        assert!(validate_gif_threshold(0.0).is_ok());
        assert!(validate_gif_threshold(0.05).is_ok());
        assert!(validate_gif_threshold(f32::NAN).is_err());
        assert!(validate_gif_threshold(f32::INFINITY).is_err());
        assert!(validate_gif_threshold(-0.1).is_err());
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridConfig {
    #[serde(default)]
    pub model_path: String, // path to weights / GOZ1 pack as needed by your runner
    /// Hidden / embedding size (Grok-1 uses 6144).
    #[serde(default = "default_grok_embedding_dim")]
    pub embedding_dim: usize,
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_top_k_experts")]
    pub top_k_experts: usize,
    #[serde(default = "default_snn_steps")]
    pub snn_steps: usize,
    #[serde(default)]
    pub projection_mode: ProjectionMode,
    #[serde(default)]
    pub execution_mode: ExecutionMode,
}

fn default_grok_embedding_dim() -> usize {
    GROK1_HIDDEN_DIM
}
fn default_num_experts() -> usize {
    8
}
fn default_top_k_experts() -> usize {
    2
}
fn default_snn_steps() -> usize {
    4
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            embedding_dim: GROK1_HIDDEN_DIM,
            num_experts: 8,
            top_k_experts: 2,
            snn_steps: 4,
            projection_mode: ProjectionMode::SpikingTernary,
            execution_mode: ExecutionMode::SpikingSim,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    #[default]
    SpikingTernary, // your main mode
                    // add others later if needed
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    #[default]
    SpikingSim, // GIF + ternary
    DenseSim, // for comparison
}

#[derive(Clone, Debug)]
pub struct HybridOutput {
    pub spike_train: Vec<Vec<usize>>,
    pub embedding: Vec<f32>,
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
}

/// Grok-1 hidden size (`hidden_size` / `embedding_length` in model cards).
pub const GROK1_HIDDEN_DIM: usize = 6144;

/// Grok-1 vocabulary size (`vocab_size` in HF `config.json`).
pub const GROK1_VOCAB_SIZE: usize = 131_072;

/// Grok-1 xai-dissect baseline inventory tensor counts (canonical fixture / upstream scan).
pub const GROK1_TENSOR_TOTAL: usize = 770;
pub const GROK1_TENSOR_F32: usize = 322;
pub const GROK1_TENSOR_INT8: usize = 448;
pub const GROK1_TENSOR_QUANT: usize = 448;
pub const GROK1_TENSOR_TOTAL_ELEMENTS: u64 = 315_684_820_992;
pub const GROK1_TENSOR_TOTAL_BYTES: u64 = 318_114_914_304;

/// One slot in a Grok-1 transformer block.
///
/// **This is the single source of truth for the 12-slot block layout.** It used
/// to be hardcoded three times — `src/artifact.rs` (`push_block_entries`),
/// `src/reports/detector.rs` (`exemplar_block_tensors` and the kind counts),
/// and `src/core/grok1_data.rs` — and the copies had already drifted: slots 00
/// and 02 were named `moe_expert.unresolved` in two of them while
/// `dissect/grok-1/structural-manifest.json` and `grok1_data.rs` had long since
/// resolved them to `.gate` and `.up`. `artifact.rs` even shipped a standing
/// warning about a question the manifest had already answered (GH #106).
///
/// The `dtype` spelling deliberately stays per-consumer: `artifact.rs` emits
/// `"int8"` into `artifact.index.json` while `grok1_data.rs` uses `"i8"`. Both
/// are user-visible, so [`BlockSlot::dtype_artifact`] and
/// [`BlockSlot::dtype_inventory`] serve them from the one `is_int8` flag rather
/// than silently normalizing one into the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSlot {
    /// Slot index within the block, 0..=11.
    pub slot: usize,
    /// Structural kind, e.g. `"moe_expert.gate"` or `"attn_proj_i8.narrow"`.
    pub kind: &'static str,
    /// `true` for the int8 attention/expert tensors, `false` for f32 norms and routers.
    pub is_int8: bool,
    /// Bytes for **one** tensor in this slot (not the per-kind aggregate).
    pub bytes: u64,
    /// Tensor shape.
    pub shape: &'static [usize],
}

impl BlockSlot {
    /// dtype as spelled in `artifact.index.json`.
    pub const fn dtype_artifact(&self) -> &'static str {
        if self.is_int8 { "int8" } else { "f32" }
    }

    /// dtype as spelled by the core inventory.
    pub const fn dtype_inventory(&self) -> &'static str {
        if self.is_int8 { "i8" } else { "f32" }
    }

    /// `true` when this slot is preserve-tier (routers and norms).
    pub const fn is_preserve(&self) -> bool {
        !self.is_int8
    }
}

const GROK1_EXPERT_BYTES: u64 = 1_610_612_736;
const GROK1_ATTN_MODEL_WIDTH_TENSOR_BYTES: u64 = 37_748_736;
const GROK1_ATTN_NARROW_TENSOR_BYTES: u64 = 6_291_456;
const GROK1_BLOCK_NORM_TENSOR_BYTES: u64 = 24_576;
const GROK1_ROUTER_TENSOR_BYTES: u64 = 196_608;

const EXPERT_GATE_SHAPE: &[usize] = &[8, GROK1_HIDDEN_DIM, 32_768];
const EXPERT_DOWN_SHAPE: &[usize] = &[8, 32_768, GROK1_HIDDEN_DIM];
const ATTN_NARROW_SHAPE: &[usize] = &[GROK1_HIDDEN_DIM, 1024];
const ATTN_MODEL_WIDTH_SHAPE: &[usize] = &[GROK1_HIDDEN_DIM, GROK1_HIDDEN_DIM];
const BLOCK_NORM_SHAPE: &[usize] = &[GROK1_HIDDEN_DIM];
const ROUTER_SHAPE: &[usize] = &[GROK1_HIDDEN_DIM, 8];

/// The 12 slots every Grok-1 block carries, in slot order.
///
/// Slots 00/01/02 are the MoE expert projections, 03..=06 the attention
/// projections, 07..=10 the block norms, and 11 the router.
pub const GROK1_BLOCK_SLOTS: [BlockSlot; 12] = [
    BlockSlot {
        slot: 0,
        kind: "moe_expert.gate",
        is_int8: true,
        bytes: GROK1_EXPERT_BYTES,
        shape: EXPERT_GATE_SHAPE,
    },
    BlockSlot {
        slot: 1,
        kind: "moe_expert.down",
        is_int8: true,
        bytes: GROK1_EXPERT_BYTES,
        shape: EXPERT_DOWN_SHAPE,
    },
    BlockSlot {
        slot: 2,
        kind: "moe_expert.up",
        is_int8: true,
        bytes: GROK1_EXPERT_BYTES,
        shape: EXPERT_GATE_SHAPE,
    },
    BlockSlot {
        slot: 3,
        kind: "attn_proj_i8.narrow",
        is_int8: true,
        bytes: GROK1_ATTN_NARROW_TENSOR_BYTES,
        shape: ATTN_NARROW_SHAPE,
    },
    BlockSlot {
        slot: 4,
        kind: "attn_proj_i8.model_width",
        is_int8: true,
        bytes: GROK1_ATTN_MODEL_WIDTH_TENSOR_BYTES,
        shape: ATTN_MODEL_WIDTH_SHAPE,
    },
    BlockSlot {
        slot: 5,
        kind: "attn_proj_i8.model_width",
        is_int8: true,
        bytes: GROK1_ATTN_MODEL_WIDTH_TENSOR_BYTES,
        shape: ATTN_MODEL_WIDTH_SHAPE,
    },
    BlockSlot {
        slot: 6,
        kind: "attn_proj_i8.narrow",
        is_int8: true,
        bytes: GROK1_ATTN_NARROW_TENSOR_BYTES,
        shape: ATTN_NARROW_SHAPE,
    },
    BlockSlot {
        slot: 7,
        kind: "block_norm",
        is_int8: false,
        bytes: GROK1_BLOCK_NORM_TENSOR_BYTES,
        shape: BLOCK_NORM_SHAPE,
    },
    BlockSlot {
        slot: 8,
        kind: "block_norm",
        is_int8: false,
        bytes: GROK1_BLOCK_NORM_TENSOR_BYTES,
        shape: BLOCK_NORM_SHAPE,
    },
    BlockSlot {
        slot: 9,
        kind: "block_norm",
        is_int8: false,
        bytes: GROK1_BLOCK_NORM_TENSOR_BYTES,
        shape: BLOCK_NORM_SHAPE,
    },
    BlockSlot {
        slot: 10,
        kind: "block_norm",
        is_int8: false,
        bytes: GROK1_BLOCK_NORM_TENSOR_BYTES,
        shape: BLOCK_NORM_SHAPE,
    },
    BlockSlot {
        slot: 11,
        kind: "router",
        is_int8: false,
        bytes: GROK1_ROUTER_TENSOR_BYTES,
        shape: ROUTER_SHAPE,
    },
];
