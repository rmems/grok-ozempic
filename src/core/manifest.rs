//! xai-dissect manifest ingestion (schema v1).
//!
//! `grok-ozempic` consumes JSON manifests produced by the upstream
//! [`xai-dissect`](https://github.com/rmems/xai-dissect) repository to drive
//! tensor selection and per-tensor precision policy. The manifest is the
//! single contract between the two repositories; `grok-ozempic` does **not**
//! depend on `xai-dissect` as a runtime crate.
//!
//! See `docs/dissect-manifest.md` for the full schema description.
//!
//! Phase 1 (this module) is **ingestion only**. The batch pipeline in
//! [`crate::core::stream`] is not rewired yet; selection and precision
//! seams land in a follow-up PR.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::error::{GrokOzempicError, Result};

/// Schema version understood by this loader.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Tensor name convention accepted in v1 (canonical Grok-1 naming used by
/// [`crate::core::npy::npy_stem_to_tensor_name`]).
pub const MANIFEST_NAME_CONVENTION_V1: &str = "blk.{L}.{role}.weight";

/// Top-level manifest document.
///
/// Unknown top-level fields are tolerated for forward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissectManifest {
    /// Schema family tag, e.g. `"xai-dissect.manifest"`. Informational.
    #[serde(default)]
    pub schema: String,

    /// Integer schema version. Must equal [`MANIFEST_SCHEMA_VERSION`].
    pub schema_version: u32,

    /// Model-identity block. `tensor_name_convention` is validated
    /// strictly by the loader.
    pub model: ManifestModel,

    /// Optional provenance block.
    #[serde(default)]
    pub produced_by: Option<ManifestProducedBy>,

    /// Pipeline-wide defaults applied when a tensor matches none of the
    /// typed sections.
    #[serde(default)]
    pub defaults: ManifestDefaults,

    /// Tensors that must keep their source precision (routing-critical
    /// layers, etc.). Reserved in phase 1; not yet consumed by the
    /// pipeline.
    #[serde(default)]
    pub preserve: Vec<PreserveEntry>,

    /// Tensors that must be emitted as FP16 passthrough.
    #[serde(default)]
    pub fp16: Vec<Fp16Entry>,

    /// Tensors ranked as candidates for ternary / SAAQ-oriented
    /// compression.
    #[serde(default)]
    pub ternary_candidates: Vec<TernaryCandidate>,

    /// Optional structural metadata (block/expert counts). Advisory in v1.
    #[serde(default)]
    pub blocks: Vec<ManifestBlock>,
}

/// Model identity carried in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestModel {
    pub family: String,
    #[serde(default)]
    pub source: String,
    pub tensor_name_convention: String,
}

/// Provenance block (informational).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestProducedBy {
    #[serde(default)]
    pub tool: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub commit: Option<String>,
}

/// Pipeline-wide defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManifestDefaults {
    /// Default precision tier. Free-form string in the manifest; mapping
    /// into [`crate::types::TensorPrecision`] happens in the selection
    /// seam (phase 2).
    #[serde(default)]
    pub precision: Option<String>,

    /// Default GIF threshold multiplier. Optional.
    #[serde(default)]
    pub gif_threshold: Option<f32>,
}

/// An entry in the `preserve` list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreserveEntry {
    /// Exact tensor name or simple glob (`*` matches one or more dotted
    /// segments).
    pub name: String,
    #[serde(default)]
    pub reason: Option<String>,
}

/// An entry in the `fp16` list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp16Entry {
    pub name: String,
    #[serde(default)]
    pub reason: Option<String>,
}

/// An entry in `ternary_candidates`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryCandidate {
    pub name: String,
    /// Optional rank hint in `[0, 1]`. Higher = stronger candidate for
    /// aggressive ternary compression.
    #[serde(default)]
    pub rank: Option<f32>,
    /// Optional per-tensor override of the global GIF threshold.
    #[serde(default)]
    pub gif_threshold: Option<f32>,
}

/// Advisory block metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestBlock {
    pub index: u32,
    #[serde(default)]
    pub experts: Option<u32>,
    #[serde(default)]
    pub role: Option<String>,
}

/// Load and validate a manifest from disk.
///
/// # Errors
///
/// Returns a typed [`GrokOzempicError`] variant rather than a best-effort
/// parse if:
///
/// - the file cannot be read ([`GrokOzempicError::ManifestIo`]),
/// - the JSON is malformed ([`GrokOzempicError::ManifestParse`]),
/// - `schema_version` is not [`MANIFEST_SCHEMA_VERSION`]
///   ([`GrokOzempicError::ManifestSchemaVersion`]),
/// - `model.tensor_name_convention` is not
///   [`MANIFEST_NAME_CONVENTION_V1`]
///   ([`GrokOzempicError::ManifestNameConventionMismatch`]).
///
/// Unknown top-level fields are tolerated.
pub fn load_manifest(path: &Path) -> Result<DissectManifest> {
    let bytes = fs::read(path).map_err(|e| GrokOzempicError::ManifestIo {
        path: path.display().to_string(),
        source: e,
    })?;

    let manifest: DissectManifest =
        serde_json::from_slice(&bytes).map_err(|e| GrokOzempicError::ManifestParse {
            path: path.display().to_string(),
            source: e,
        })?;

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(GrokOzempicError::ManifestSchemaVersion {
            got: manifest.schema_version,
            expected: MANIFEST_SCHEMA_VERSION,
        });
    }

    if manifest.model.tensor_name_convention != MANIFEST_NAME_CONVENTION_V1 {
        return Err(GrokOzempicError::ManifestNameConventionMismatch {
            got: manifest.model.tensor_name_convention.clone(),
            expected: MANIFEST_NAME_CONVENTION_V1.to_string(),
        });
    }

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, contents: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("grok-ozempic-manifest-test-{name}.json"));
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path
    }

    fn valid_manifest_json() -> &'static str {
        r#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 1,
            "model": {
                "family": "grok-1",
                "source": "xai-org/grok-1",
                "tensor_name_convention": "blk.{L}.{role}.weight"
            },
            "produced_by": {
                "tool": "xai-dissect",
                "version": "0.1.0"
            },
            "defaults": {
                "precision": "ternary_snn",
                "gif_threshold": 0.05
            },
            "preserve": [
                { "name": "blk.*.attn_router.weight", "reason": "routing-critical" }
            ],
            "fp16": [
                { "name": "token_embd.weight" }
            ],
            "ternary_candidates": [
                { "name": "blk.0.ffn_up.weight", "rank": 0.98, "gif_threshold": 0.04 }
            ],
            "blocks": [
                { "index": 0, "experts": 8, "role": "moe" }
            ]
        }"#
    }

    #[test]
    fn loads_valid_v1_manifest() {
        let path = write_tmp("valid", valid_manifest_json());
        let manifest = load_manifest(&path).expect("valid manifest should load");
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.model.family, "grok-1");
        assert_eq!(manifest.preserve.len(), 1);
        assert_eq!(manifest.fp16.len(), 1);
        assert_eq!(manifest.ternary_candidates.len(), 1);
        assert_eq!(manifest.blocks.len(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_manifest_io_error() {
        let mut path = std::env::temp_dir();
        path.push("grok-ozempic-manifest-test-does-not-exist.json");
        let _ = fs::remove_file(&path);
        let err = load_manifest(&path).unwrap_err();
        assert!(
            matches!(err, GrokOzempicError::ManifestIo { .. }),
            "expected ManifestIo, got {err:?}"
        );
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let json = r#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 2,
            "model": {
                "family": "grok-1",
                "tensor_name_convention": "blk.{L}.{role}.weight"
            }
        }"#;
        let path = write_tmp("version", json);
        let err = load_manifest(&path).unwrap_err();
        assert!(
            matches!(
                err,
                GrokOzempicError::ManifestSchemaVersion { got: 2, expected: 1 }
            ),
            "expected ManifestSchemaVersion, got {err:?}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn mismatched_name_convention_hard_fails() {
        let json = r#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 1,
            "model": {
                "family": "grok-1",
                "tensor_name_convention": "model.layers.{L}.weight"
            }
        }"#;
        let path = write_tmp("convention", json);
        let err = load_manifest(&path).unwrap_err();
        assert!(
            matches!(
                err,
                GrokOzempicError::ManifestNameConventionMismatch { .. }
            ),
            "expected ManifestNameConventionMismatch, got {err:?}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn unknown_top_level_field_is_tolerated() {
        let json = r#"{
            "schema": "xai-dissect.manifest",
            "schema_version": 1,
            "future_field": { "hello": "world" },
            "model": {
                "family": "grok-1",
                "tensor_name_convention": "blk.{L}.{role}.weight"
            }
        }"#;
        let path = write_tmp("unknown", json);
        let manifest = load_manifest(&path).expect("unknown fields should be tolerated");
        assert_eq!(manifest.schema_version, 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn in_tree_grok1_baseline_loads() {
        // Guard: the committed reference manifest must always parse against
        // the current loader. It is non-authoritative (xai-dissect is the
        // source of truth) but it is the crate's own bootstrapping example.
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("dissect/grok-1/baseline.json");
        let manifest = load_manifest(&path).expect("baseline must load");
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.model.family, "grok-1");
        assert_eq!(
            manifest.model.tensor_name_convention,
            MANIFEST_NAME_CONVENTION_V1
        );
    }

    #[test]
    fn malformed_json_is_parse_error() {
        let path = write_tmp("malformed", "{ not valid json");
        let err = load_manifest(&path).unwrap_err();
        assert!(
            matches!(err, GrokOzempicError::ManifestParse { .. }),
            "expected ManifestParse, got {err:?}"
        );
        let _ = fs::remove_file(&path);
    }
}
