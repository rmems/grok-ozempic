//! Out-of-core streaming quantization engine
//!
//! 1. **Manifest pass** — scans safetensors shards or `.npy` files and records
//!    only `(path, name, shape, dtype, router?)` (no weight payloads).
//! 2. **Header pass** — writes **GOZ1** metadata + tensor table with placeholder
//!    offsets via [`PackStreamWriter::begin`](crate::core::weight_pack::PackStreamWriter::begin).
//! 3. **Data pass** — for each manifest row, memory-maps the source tensor,
//!    quantizes **one tensor at a time**, and streams bytes with
//!    [`PackStreamWriter::write_tensor_data`](crate::core::weight_pack::PackStreamWriter::write_tensor_data).
//!
//! **JAX / Flax** checkpoints are not read directly; export tensors to **`.npy`**
//! (or **safetensors**) first. Python **pickle** is not supported.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

use half::f16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;

use crate::{
    core::{
        manifest::{embedded_grok1_baseline, load_manifest, DissectManifest},
        npy::{npy_stem_to_tensor_name, MmapNpy, NpyDtype},
        precision::decide as precision_decide,
        quantizer::{convert_f32_to_f16_bytes, passthrough_f16, quantize_f16, quantize_f32},
        selection::{classify, TensorClass},
        weight_pack::{
            PackMetaValue, PackStreamWriter, PackTensorHeader, TENSOR_F16, TENSOR_TERNARY,
        },
    },
    error::{GrokOzempicError, Result},
    types::{QuantizationConfig, QuantizationInputFormat, TensorPrecision},
};

// Grok-1 (JAX) architecture hints embedded in GOZ1 metadata — confirm against
// https://huggingface.co/xai-org/grok-1 `config.json` when updating.
const GROK1_CONTEXT_LENGTH: u32 = 8192;
const GROK1_EMBEDDING_LENGTH: u32 = 6144;
const GROK1_FEED_FORWARD_LENGTH: u32 = 32768;
const GROK1_ATTENTION_HEAD_COUNT: u32 = 48;
const GROK1_ATTENTION_HEAD_COUNT_KV: u32 = 8;
const GROK1_BLOCK_COUNT: u32 = 64;
/// MoE experts **per layer** (matches HF Grok-1 and [`crate::types::HybridConfig`] defaults).
const GROK1_EXPERT_COUNT: u32 = 8;

/// Append Grok-1 model shape hints into GOZ1 metadata (for your own loaders / JAX tooling).
pub fn append_grok1_arch_metadata(meta: &mut BTreeMap<String, PackMetaValue>) {
    meta.insert(
        "grok1.architecture".into(),
        PackMetaValue::Str("grok-1".into()),
    );
    meta.insert(
        "grok1.context_length".into(),
        PackMetaValue::U32(GROK1_CONTEXT_LENGTH),
    );
    meta.insert(
        "grok1.embedding_length".into(),
        PackMetaValue::U32(GROK1_EMBEDDING_LENGTH),
    );
    meta.insert(
        "grok1.feed_forward_length".into(),
        PackMetaValue::U32(GROK1_FEED_FORWARD_LENGTH),
    );
    meta.insert(
        "grok1.attention.head_count".into(),
        PackMetaValue::U32(GROK1_ATTENTION_HEAD_COUNT),
    );
    meta.insert(
        "grok1.attention.head_count_kv".into(),
        PackMetaValue::U32(GROK1_ATTENTION_HEAD_COUNT_KV),
    );
    meta.insert(
        "grok1.block_count".into(),
        PackMetaValue::U32(GROK1_BLOCK_COUNT),
    );
    meta.insert(
        "grok1.expert_count".into(),
        PackMetaValue::U32(GROK1_EXPERT_COUNT),
    );
}

// ---------------------------------------------------------------------------
// Manifest resolution & deprecation warning
// ---------------------------------------------------------------------------

/// Environment variable consulted when [`QuantizationConfig::manifest_path`]
/// is `None`.
pub const MANIFEST_ENV_VAR: &str = "GROK_OZEMPIC_MANIFEST";

/// Resolve the xai-dissect manifest that governs this `run_quantization`
/// call.
///
/// Precedence (first hit wins, errors bubble; no silent fallthrough):
///
/// 1. [`QuantizationConfig::manifest_path`] (explicit caller override).
/// 2. `GROK_OZEMPIC_MANIFEST` environment variable.
/// 3. Embedded Grok-1 baseline via [`embedded_grok1_baseline`].
/// 4. `None` — selection falls back to the legacy heuristic in
///    [`crate::core::selection::classify`].
fn resolve_manifest(config: &QuantizationConfig) -> Result<Option<DissectManifest>> {
    if let Some(path) = &config.manifest_path {
        return Ok(Some(load_manifest(path)?));
    }
    if let Ok(env_path) = std::env::var(MANIFEST_ENV_VAR) {
        if !env_path.is_empty() {
            let p = std::path::PathBuf::from(env_path);
            return Ok(Some(load_manifest(&p)?));
        }
    }
    // Embedded baseline: clone once so callers own a DissectManifest.
    // The OnceLock inside manifest.rs keeps the parse one-shot.
    Ok(Some(embedded_grok1_baseline()?.clone()))
}

/// Return `true` if a deprecation warning about `router_patterns`
/// alongside a manifest should be emitted.
///
/// Exposed (crate-private) so tests can assert the condition without
/// capturing stderr.
pub(crate) fn should_warn_router_patterns_deprecated(
    manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> bool {
    manifest.is_some() && !config.router_patterns.is_empty()
}

fn maybe_warn_router_patterns_deprecated(
    manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) {
    if should_warn_router_patterns_deprecated(manifest, config) {
        eprintln!(
            "grok-ozempic: router_patterns is set alongside an xai-dissect manifest; \
             the manifest wins and router_patterns is ignored (deprecated)."
        );
    }
}

// ---------------------------------------------------------------------------
// Source dtype (safetensors + npy)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceDtype {
    F32,
    F16,
    BF16,
    Other,
}

fn parse_safetensors_dtype(dt: safetensors::Dtype) -> SourceDtype {
    match dt {
        safetensors::Dtype::F32 => SourceDtype::F32,
        safetensors::Dtype::F16 => SourceDtype::F16,
        safetensors::Dtype::BF16 => SourceDtype::BF16,
        _ => SourceDtype::Other,
    }
}

fn npy_dtype_to_source(dt: NpyDtype) -> SourceDtype {
    match dt {
        NpyDtype::F32 => SourceDtype::F32,
        NpyDtype::F16 => SourceDtype::F16,
        NpyDtype::BF16 => SourceDtype::BF16,
        NpyDtype::Other => SourceDtype::Other,
    }
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ManifestEntry {
    source_path: PathBuf,
    tensor_name: String,
    shape: Vec<u64>,
    precision: TensorPrecision,
    gif_threshold: f32,
}

impl ManifestEntry {
    /// True when this tensor goes through the FP16 passthrough path.
    ///
    /// TODO(phase-3): [`TensorPrecision::Preserve`] is currently aliased
    /// to FP16 passthrough here. This is the single site of the
    /// transitional alias; removal tracked in issue #6.
    fn uses_fp16_passthrough(&self) -> bool {
        matches!(
            self.precision,
            TensorPrecision::Fp16 | TensorPrecision::Preserve
        )
    }
}

/// Diagnostic statistics emitted after processing each shard / source file.
#[derive(Debug, Default)]
pub struct ShardStats {
    pub shard_path: PathBuf,
    pub tensors_ternary: usize,
    pub tensors_fp16: usize,
    pub tensors_skipped: usize,
    pub avg_sparsity: f32,
}

/// Run the full out-of-core quantization pipeline described by `config`.
pub fn run_quantization(config: &QuantizationConfig) -> Result<Vec<ShardStats>> {
    let dissect_manifest = resolve_manifest(config)?;
    maybe_warn_router_patterns_deprecated(dissect_manifest.as_ref(), config);

    let manifest = match config.input_format {
        QuantizationInputFormat::Safetensors => {
            let shards = collect_safetensor_shards(&config.input_dir)?;
            if shards.is_empty() {
                return Err(GrokOzempicError::InvalidConfig(format!(
                    "no .safetensors files found in '{}'",
                    config.input_dir
                )));
            }
            build_manifest_safetensors(&shards, dissect_manifest.as_ref(), config)?
        }
        QuantizationInputFormat::NpyDir => {
            let paths = collect_npy_files(&config.input_dir)?;
            if paths.is_empty() {
                return Err(GrokOzempicError::InvalidConfig(format!(
                    "no .npy files found in '{}'",
                    config.input_dir
                )));
            }
            build_manifest_npy(&paths, dissect_manifest.as_ref(), config)?
        }
    };

    if manifest.is_empty() {
        return Err(GrokOzempicError::InvalidConfig(
            "no quantizable tensors found (all skipped as unsupported dtype?)".into(),
        ));
    }

    let headers: Vec<PackTensorHeader> = manifest
        .iter()
        .map(|e| PackTensorHeader {
            name: e.tensor_name.clone(),
            shape: e.shape.clone(),
            // TODO(phase-3): Preserve currently shares TENSOR_F16 with
            // Fp16. Once source-dtype passthrough lands, introduce a
            // dedicated TENSOR_PRESERVE constant. Tracked in issue #6.
            tensor_type: if e.uses_fp16_passthrough() {
                TENSOR_F16
            } else {
                TENSOR_TERNARY
            },
        })
        .collect();

    let mut metadata: BTreeMap<String, PackMetaValue> = BTreeMap::new();
    metadata.insert(
        "oz.name".into(),
        PackMetaValue::Str("grok-ozempic".into()),
    );
    metadata.insert(
        "oz.quantization_version".into(),
        PackMetaValue::U32(1),
    );
    metadata.insert(
        "oz.gif_threshold".into(),
        PackMetaValue::Str(config.gif_threshold.to_string()),
    );
    append_grok1_arch_metadata(&mut metadata);

    let out_file = File::create(&config.output_path).map_err(GrokOzempicError::Io)?;
    let mut out_writer = BufWriter::new(out_file);

    let mut stream = PackStreamWriter::begin(&mut out_writer, &metadata, &headers)?;

    let mut all_stats: Vec<ShardStats> = Vec::new();
    let mut current_path: Option<PathBuf> = None;
    let mut shard_stats = ShardStats::default();
    let mut sparsity_sum = 0.0f32;
    let mut sparsity_n = 0usize;

    for entry in &manifest {
        if current_path.as_ref() != Some(&entry.source_path) {
            if current_path.is_some() {
                if sparsity_n > 0 {
                    shard_stats.avg_sparsity = sparsity_sum / sparsity_n as f32;
                }
                all_stats.push(shard_stats);
                sparsity_sum = 0.0;
                sparsity_n = 0;
            }
            shard_stats = ShardStats {
                shard_path: entry.source_path.clone(),
                ..Default::default()
            };
            current_path = Some(entry.source_path.clone());
        }

        let (packed, ternary_sparsity) = quantize_manifest_entry(entry, config)?;
        if entry.uses_fp16_passthrough() {
            shard_stats.tensors_fp16 += 1;
        } else {
            shard_stats.tensors_ternary += 1;
            if let Some(sp) = ternary_sparsity {
                sparsity_sum += sp;
                sparsity_n += 1;
            }
        }
        stream.write_tensor_data(&packed)?;
    }
    if current_path.is_some() {
        if sparsity_n > 0 {
            shard_stats.avg_sparsity = sparsity_sum / sparsity_n as f32;
        }
        all_stats.push(shard_stats);
    }

    stream.finalize()?;
    out_writer
        .flush()
        .map_err(GrokOzempicError::Io)?;

    Ok(all_stats)
}

fn quantize_manifest_entry(
    entry: &ManifestEntry,
    config: &QuantizationConfig,
) -> Result<(Vec<u8>, Option<f32>)> {
    match config.input_format {
        QuantizationInputFormat::Safetensors => quantize_safetensors_entry(entry, config),
        QuantizationInputFormat::NpyDir => quantize_npy_entry(entry, config),
    }
}

fn quantize_safetensors_entry(
    entry: &ManifestEntry,
    _config: &QuantizationConfig,
) -> Result<(Vec<u8>, Option<f32>)> {
    let file = File::open(&entry.source_path).map_err(GrokOzempicError::Io)?;
    let mmap = unsafe { MmapOptions::new().map(&file).map_err(GrokOzempicError::Io)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(GrokOzempicError::Safetensors)?;
    let view = tensors.tensor(&entry.tensor_name)?;

    let dtype = parse_safetensors_dtype(view.dtype());
    if dtype == SourceDtype::Other {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "tensor {} has unsupported dtype",
            entry.tensor_name
        )));
    }

    if entry.uses_fp16_passthrough() {
        let fp16_bytes = router_fp16_bytes(dtype, view.data())?;
        Ok((fp16_bytes, None))
    } else {
        let qt = match dtype {
            SourceDtype::F32 => {
                let f32_slice = bytemuck_cast_f32(view.data());
                quantize_f32(f32_slice, entry.gif_threshold)
            }
            SourceDtype::F16 => {
                let f16_slice: &[f16] = bytemuck_cast_f16(view.data());
                quantize_f16(f16_slice, entry.gif_threshold)
            }
            SourceDtype::BF16 => {
                let f32_vals = bf16_bytes_to_f32(view.data());
                quantize_f32(&f32_vals, entry.gif_threshold)
            }
            SourceDtype::Other => unreachable!(),
        };
        let sp = qt.sparsity;
        Ok((qt.packed, Some(sp)))
    }
}

fn quantize_npy_entry(
    entry: &ManifestEntry,
    _config: &QuantizationConfig,
) -> Result<(Vec<u8>, Option<f32>)> {
    let npy = MmapNpy::map_path(&entry.source_path)?;
    let dtype = npy_dtype_to_source(npy.dtype());
    if dtype == SourceDtype::Other {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "npy {} has unsupported descr",
            entry.source_path.display()
        )));
    }
    let raw = npy.data();

    if entry.uses_fp16_passthrough() {
        let fp16_bytes = router_fp16_bytes(dtype, raw)?;
        Ok((fp16_bytes, None))
    } else {
        let qt = match dtype {
            SourceDtype::F32 => {
                let f32_slice = bytemuck_cast_f32(raw);
                quantize_f32(f32_slice, entry.gif_threshold)
            }
            SourceDtype::F16 => {
                let f16_slice: &[f16] = bytemuck_cast_f16(raw);
                quantize_f16(f16_slice, entry.gif_threshold)
            }
            SourceDtype::BF16 => {
                let f32_vals = bf16_bytes_to_f32(raw);
                quantize_f32(&f32_vals, entry.gif_threshold)
            }
            SourceDtype::Other => unreachable!(),
        };
        let sp = qt.sparsity;
        Ok((qt.packed, Some(sp)))
    }
}

fn router_fp16_bytes(dtype: SourceDtype, raw: &[u8]) -> Result<Vec<u8>> {
    let b = match dtype {
        SourceDtype::F16 => {
            let f16_slice: &[f16] = bytemuck_cast_f16(raw);
            passthrough_f16(f16_slice)
        }
        SourceDtype::F32 => {
            let f32_slice = bytemuck_cast_f32(raw);
            convert_f32_to_f16_bytes(f32_slice)
        }
        SourceDtype::BF16 => {
            let f32_vals = bf16_bytes_to_f32(raw);
            convert_f32_to_f16_bytes(&f32_vals)
        }
        SourceDtype::Other => unreachable!(),
    };
    Ok(b)
}

fn classify_and_decide(
    name: &str,
    dissect_manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> Result<(TensorClass, TensorPrecision, f32)> {
    // selection.rs is the single source of truth for classification.
    // When a manifest is supplied, router_patterns is ignored (the
    // deprecation warning above already informed the caller).
    let legacy_patterns: &[String] = if dissect_manifest.is_some() {
        &[]
    } else {
        &config.router_patterns
    };
    let class = classify(name, dissect_manifest, legacy_patterns);
    let (precision, gif_threshold) = precision_decide(&class, dissect_manifest, config)?;
    Ok((class, precision, gif_threshold))
}

fn build_manifest_safetensors(
    shards: &[PathBuf],
    dissect_manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> Result<Vec<ManifestEntry>> {
    let mut v = Vec::new();
    for shard in shards {
        let file = File::open(shard).map_err(GrokOzempicError::Io)?;
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(GrokOzempicError::Io)? };
        let tensors = SafeTensors::deserialize(&mmap).map_err(GrokOzempicError::Safetensors)?;
        for (name, view) in tensors.tensors() {
            let dtype = parse_safetensors_dtype(view.dtype());
            if dtype == SourceDtype::Other {
                continue;
            }
            let (_class, precision, gif_threshold) =
                classify_and_decide(&name, dissect_manifest, config)?;
            let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
            v.push(ManifestEntry {
                source_path: shard.clone(),
                tensor_name: name,
                shape,
                precision,
                gif_threshold,
            });
        }
    }
    Ok(v)
}

fn build_manifest_npy(
    paths: &[PathBuf],
    dissect_manifest: Option<&DissectManifest>,
    config: &QuantizationConfig,
) -> Result<Vec<ManifestEntry>> {
    let mut v = Vec::new();
    for path in paths {
        let npy = MmapNpy::map_path(path)?;
        let dtype = npy_dtype_to_source(npy.dtype());
        if dtype == SourceDtype::Other {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                GrokOzempicError::InvalidConfig(format!("bad npy filename: {}", path.display()))
            })?;
        let tensor_name = npy_stem_to_tensor_name(stem);
        let (_class, precision, gif_threshold) =
            classify_and_decide(&tensor_name, dissect_manifest, config)?;
        let shape: Vec<u64> = npy.shape().iter().map(|&d| d as u64).collect();
        v.push(ManifestEntry {
            source_path: path.clone(),
            tensor_name,
            shape,
            precision,
            gif_threshold,
        });
    }
    Ok(v)
}

fn collect_safetensor_shards(dir: &str) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(GrokOzempicError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    paths.sort();
    Ok(paths)
}

fn collect_npy_files(dir: &str) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(GrokOzempicError::Io)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "npy"))
        .collect();
    paths.sort();
    Ok(paths)
}

fn bytemuck_cast_f32(raw: &[u8]) -> &[f32] {
    assert_eq!(raw.len() % 4, 0, "f32 data length must be a multiple of 4");
    unsafe {
        std::slice::from_raw_parts(raw.as_ptr().cast::<f32>(), raw.len() / 4)
    }
}

fn bytemuck_cast_f16(raw: &[u8]) -> &[f16] {
    assert_eq!(raw.len() % 2, 0, "f16 data length must be a multiple of 2");
    unsafe {
        std::slice::from_raw_parts(raw.as_ptr().cast::<f16>(), raw.len() / 2)
    }
}

fn bf16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    assert_eq!(raw.len() % 2, 0, "bf16 data length must be a multiple of 2");
    raw.chunks_exact(2)
        .map(|chunk| {
            let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let f32_bits = (bf16_bits as u32) << 16;
            f32::from_bits(f32_bits)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Router-classification tests moved to src/core/selection.rs; this
    // file no longer owns classification logic.

    #[test]
    fn deprecation_warning_fires_when_both_present() {
        use crate::core::manifest::{
            DissectManifest, ManifestDefaults, ManifestModel, MANIFEST_NAME_CONVENTION_V1,
            MANIFEST_SCHEMA_VERSION,
        };
        let manifest = DissectManifest {
            schema: "xai-dissect.manifest".into(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            model: ManifestModel {
                family: "grok-1".into(),
                source: "".into(),
                tensor_name_convention: MANIFEST_NAME_CONVENTION_V1.into(),
            },
            produced_by: None,
            defaults: ManifestDefaults::default(),
            preserve: vec![],
            fp16: vec![],
            ternary_candidates: vec![],
            blocks: vec![],
        };

        let config_with_patterns = QuantizationConfig {
            router_patterns: vec!["legacy_router".into()],
            ..Default::default()
        };
        assert!(should_warn_router_patterns_deprecated(
            Some(&manifest),
            &config_with_patterns,
        ));

        let config_empty = QuantizationConfig::default();
        assert!(!should_warn_router_patterns_deprecated(
            Some(&manifest),
            &config_empty,
        ));

        assert!(!should_warn_router_patterns_deprecated(
            None,
            &config_with_patterns,
        ));
    }

    #[test]
    fn bf16_bytes_to_f32_one() {
        let raw = [0x80u8, 0x3F];
        let result = bf16_bytes_to_f32(&raw);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1.0f32);
    }

    #[test]
    fn bf16_bytes_to_f32_zero() {
        let raw = [0x00u8, 0x00];
        let result = bf16_bytes_to_f32(&raw);
        assert_eq!(result[0], 0.0f32);
    }

    #[test]
    fn bytemuck_cast_f32_basic() {
        let floats = [1.0f32, -2.0f32, 3.5f32];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let back = bytemuck_cast_f32(&bytes);
        assert_eq!(back, floats);
    }

    #[test]
    fn collect_shards_nonexistent_dir() {
        let result = collect_safetensor_shards("/tmp/nonexistent_grok_ozempic_dir_xyz");
        assert!(result.is_err());
    }
}
