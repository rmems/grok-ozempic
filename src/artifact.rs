use crate::core::manifest::{DissectManifest, parse_manifest_bytes};
use crate::core::stream::{GROK1_BLOCK_COUNT, GROK1_EXPERT_COUNT};
use crate::error::{GrokOzempicError, Result};
use crate::types::{GROK1_HIDDEN_DIM, GROK1_TENSOR_TOTAL, GROK1_TENSOR_TOTAL_BYTES};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

pub const GROK1_ARTIFACT_FORMAT: &str = "saaq-g1-v0";
pub const GROK1_ROUTER_SHAPE: [usize; 2] = [GROK1_HIDDEN_DIM, GROK1_EXPERT_COUNT as usize];
pub const GROK1_ROUTER_ORIENTATION: &str = "d_model_to_experts";
pub const GROK1_EXPERT_FAMILIES_PER_BLOCK: usize = 3;
pub const GROK1_BLOCK_TENSORS: usize = 12;
pub const GROK1_NORMS_PER_BLOCK: usize = 4;
pub const GROK1_UNKNOWN_DENSE_PER_BLOCK: usize = 4;

const EMBEDDING_BYTES: u64 = 3_221_225_472;
const FINAL_NORM_BYTES: u64 = 24_576;
const EXPERT_BYTES: u64 = 1_610_612_736;
const ATTN_MODEL_WIDTH_BYTES: u64 = 37_748_736;
const ATTN_NARROW_BYTES: u64 = 6_291_456;
const NORM_BYTES: u64 = 24_576;
const ROUTER_BYTES: u64 = 196_608;
const CHECKSUMS_FILE: &str = "checksums.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactIndex {
    pub schema: String,
    pub schema_version: u32,
    pub format: String,
    pub model_family: String,
    pub mode: String,
    pub dry_run: bool,
    pub tensor_count: usize,
    pub source_total_bytes: u64,
    pub artifact_total_bytes: u64,
    pub router_count: usize,
    pub entries: Vec<ArtifactIndexEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactIndexEntry {
    pub source_tensor_name: String,
    pub structural_name: String,
    pub block: Option<usize>,
    pub slot: Option<usize>,
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_len: u64,
    pub source_checksum: Option<String>,
    pub output_checksum: String,
    pub quant_policy_applied: String,
    pub artifact_path: String,
    pub artifact_offset: u64,
    pub protected: bool,
    pub protected_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarningRecord {
    pub category: String,
    pub tensor: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecord {
    pub category: String,
    pub tensor: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub status: String,
    pub source_tensor_count: usize,
    pub artifact_tensor_count: usize,
    pub router_count: usize,
    pub protected_router_violations: usize,
    pub protected_norm_violations: usize,
    pub expert_association_count: usize,
    pub unknown_unresolved_warning_count: usize,
    pub checksum_coverage: String,
    pub source_total_bytes: u64,
    pub artifact_total_bytes: u64,
    pub byte_accounting_result: String,
    pub failures: Vec<FailureRecord>,
    pub warnings: Vec<WarningRecord>,
}

pub struct ConvertOptions<'a> {
    pub checkpoint: Option<&'a Path>,
    pub manifest: &'a Path,
    pub output_root: &'a Path,
    pub format: &'a str,
    pub protect_routers: bool,
    pub protect_norms: bool,
    pub dry_run: bool,
}

pub struct SmokeOptions<'a> {
    pub checkpoint: Option<&'a Path>,
    pub manifest: &'a Path,
    pub block: usize,
    pub include_embedding: bool,
    pub include_final_norm: bool,
    pub output_root: &'a Path,
    pub dry_run: bool,
}

pub fn validate_ingest_path(
    manifest_path: &Path,
    checkpoint: Option<&Path>,
) -> Result<DissectManifest> {
    let manifest_bytes = fs::read(manifest_path).map_err(|e| GrokOzempicError::ManifestIo {
        path: manifest_path.display().to_string(),
        source: e,
    })?;
    let manifest = parse_manifest_bytes(&manifest_bytes, &manifest_path.display().to_string())?;
    validate_grok1_manifest_identity(&manifest)?;

    if let Some(checkpoint) = checkpoint {
        if !checkpoint.exists() {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "checkpoint path does not exist: {}",
                checkpoint.display()
            )));
        }
        if !checkpoint.is_dir() {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "checkpoint path is not a directory: {}",
                checkpoint.display()
            )));
        }
        validate_checkpoint_checksums(checkpoint)?;
    }

    Ok(manifest)
}

pub fn convert_grok1(options: ConvertOptions<'_>) -> Result<ArtifactIndex> {
    let manifest = validate_ingest_path(options.manifest, options.checkpoint)?;
    validate_grok1_manifest_identity(&manifest)?;
    if options.format != GROK1_ARTIFACT_FORMAT {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "unsupported Grok-1 artifact format: got {}, expected {}",
            options.format, GROK1_ARTIFACT_FORMAT
        )));
    }
    let warnings = planned_warnings();
    let entries = planned_grok1_entries(options.protect_routers, options.protect_norms, None)?;
    validate_full_expected_entries(&entries, options.protect_routers, options.protect_norms)?;
    let index = build_index("full", options.format, options.dry_run, entries);
    write_conversion_outputs(
        options.output_root,
        options.manifest,
        &index,
        &warnings,
        options.dry_run,
    )?;
    Ok(index)
}

pub fn smoke_grok1(options: SmokeOptions<'_>) -> Result<ArtifactIndex> {
    let manifest = validate_ingest_path(options.manifest, options.checkpoint)?;
    validate_grok1_manifest_identity(&manifest)?;
    if options.block >= GROK1_BLOCK_COUNT as usize {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "smoke block {} is outside Grok-1 range 0..{}",
            options.block, GROK1_BLOCK_COUNT
        )));
    }

    let selected = SmokeSelection {
        block: options.block,
        include_embedding: options.include_embedding,
        include_final_norm: options.include_final_norm,
    };
    let entries = planned_grok1_entries(true, true, Some(selected))?;
    validate_smoke_entries(
        &entries,
        options.block,
        options.include_embedding,
        options.include_final_norm,
    )?;
    let index = build_index("smoke", GROK1_ARTIFACT_FORMAT, options.dry_run, entries);
    write_smoke_outputs(
        options.output_root,
        &index,
        &planned_warnings(),
        options.dry_run,
    )?;
    Ok(index)
}

pub fn validate_grok1_artifact(
    manifest: &Path,
    artifact_index: &Path,
    checksums: Option<&Path>,
    output_root: Option<&Path>,
    strict_router_protection: bool,
) -> Result<ValidationReport> {
    let source_manifest = validate_ingest_path(manifest, None)?;
    validate_grok1_manifest_identity(&source_manifest)?;

    let index_bytes = fs::read(artifact_index)?;
    let index: ArtifactIndex = serde_json::from_slice(&index_bytes).map_err(|e| {
        GrokOzempicError::ArtifactValidation(format!(
            "failed to parse artifact index {}: {e}",
            artifact_index.display()
        ))
    })?;

    let checksum_map = if let Some(path) = checksums {
        Some(load_checksums_file(path)?)
    } else {
        None
    };

    let expected_entries = expected_full_entry_map()?;
    let report = build_validation_report(
        &index,
        GROK1_ARTIFACT_FORMAT,
        strict_router_protection,
        true,
        Some(&expected_entries),
        checksum_map.as_ref(),
    );
    if let Some(dir) = output_root {
        write_validation_outputs(dir, &report)?;
    }
    if report.status != "PASS" {
        let mut message = format!(
            "Grok-1 artifact validation failed with {} failure(s)",
            report.failures.len()
        );
        if output_root.is_some() {
            message.push_str("; see validation.report.json");
        } else if let Some(first) = report.failures.first() {
            message.push_str(&format!(
                "; first failure [{}] {}. rerun with --output-root to write validation.report.json",
                first.category, first.message
            ));
        }
        return Err(GrokOzempicError::ArtifactValidation(format!("{message}")));
    }
    Ok(report)
}

fn validate_grok1_manifest_identity(manifest: &DissectManifest) -> Result<()> {
    if manifest.model.family != "grok-1" {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "expected manifest model.family grok-1, got {}",
            manifest.model.family
        )));
    }
    Ok(())
}

fn validate_checkpoint_checksums(checkpoint: &Path) -> Result<()> {
    let checksums = checkpoint.join(CHECKSUMS_FILE);
    if !checksums.exists() {
        return Ok(());
    }
    let checkpoint_root = checkpoint.canonicalize().map_err(|e| {
        GrokOzempicError::ArtifactValidation(format!(
            "failed to canonicalize checkpoint root {}: {e}",
            checkpoint.display()
        ))
    })?;
    let map = load_checksums_file(&checksums)?;
    for (relative, expected) in map {
        let path = resolve_checkpoint_checksum_entry_path(checkpoint, &relative)?;
        let canonical_path = path.canonicalize().map_err(|e| {
            GrokOzempicError::ArtifactValidation(format!(
                "failed to canonicalize checksum entry {}: {e}",
                path.display()
            ))
        })?;
        if !canonical_path.starts_with(&checkpoint_root) {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "checksum entry resolves outside checkpoint root: {}",
                path.display()
            )));
        }
        if !path.is_file() {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "checksum entry references missing file: {}",
                path.display()
            )));
        }
        let actual = sha256_file(&path)?;
        if normalize_checksum(&expected) != actual {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "checksum mismatch for {}: expected {}, got sha256:{}",
                path.display(),
                expected,
                actual
            )));
        }
    }
    Ok(())
}

fn load_checksums_file(path: &Path) -> Result<BTreeMap<String, String>> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| {
        GrokOzempicError::ArtifactValidation(format!(
            "failed to parse checksum map {}: {e}",
            path.display()
        ))
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn normalize_checksum(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    lower.strip_prefix("sha256:").unwrap_or(&lower).to_string()
}

fn resolve_checkpoint_checksum_entry_path(checkpoint: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "illegal path traversal or absolute path in checksum relative path: {relative}"
        )));
    }
    Ok(checkpoint.join(relative_path))
}

#[derive(Clone, Copy)]
struct SmokeSelection {
    block: usize,
    include_embedding: bool,
    include_final_norm: bool,
}

fn planned_grok1_entries(
    protect_routers: bool,
    protect_norms: bool,
    selection: Option<SmokeSelection>,
) -> Result<Vec<ArtifactIndexEntry>> {
    let mut entries = Vec::new();
    let mut offset = 0u64;

    let include_embedding = selection.map(|s| s.include_embedding).unwrap_or(true);
    if include_embedding {
        push_entry(
            &mut entries,
            &mut offset,
            TensorPlan::new(
                "embedding.slot_00.token_embedding",
                "embedding.slot_00.token_embedding",
                None,
                Some(0),
                "token_embedding",
                "f32",
                vec![131_072, GROK1_HIDDEN_DIM],
                EMBEDDING_BYTES,
                "candidate_saaq_embedding",
                false,
                None,
            ),
        );
    }

    let block_range: Vec<usize> = if let Some(selection) = selection {
        vec![selection.block]
    } else {
        (0..GROK1_BLOCK_COUNT as usize).collect()
    };

    for block in block_range {
        push_block_entries(
            &mut entries,
            &mut offset,
            block,
            protect_routers,
            protect_norms,
        );
    }

    let include_final_norm = selection.map(|s| s.include_final_norm).unwrap_or(true);
    if include_final_norm {
        push_entry(
            &mut entries,
            &mut offset,
            TensorPlan::new(
                "final_norm.slot_00.final_norm",
                "final_norm.slot_00.final_norm",
                None,
                Some(0),
                "final_norm",
                "f32",
                vec![GROK1_HIDDEN_DIM],
                FINAL_NORM_BYTES,
                "passthrough_f32_norm",
                protect_norms,
                protect_norms.then_some("final norm protected by policy"),
            ),
        );
    }

    Ok(entries)
}

fn push_block_entries(
    entries: &mut Vec<ArtifactIndexEntry>,
    offset: &mut u64,
    block: usize,
    protect_routers: bool,
    protect_norms: bool,
) {
    let prefix = format!("block_{block:03}");
    for (slot, kind, shape, bytes) in [
        (
            0,
            "moe_expert.unresolved",
            vec![8, GROK1_HIDDEN_DIM, 32_768],
            EXPERT_BYTES,
        ),
        (
            1,
            "moe_expert.down",
            vec![8, 32_768, GROK1_HIDDEN_DIM],
            EXPERT_BYTES,
        ),
        (
            2,
            "moe_expert.unresolved",
            vec![8, GROK1_HIDDEN_DIM, 32_768],
            EXPERT_BYTES,
        ),
        (
            3,
            "unknown_dense.narrow",
            vec![GROK1_HIDDEN_DIM, 1024],
            ATTN_NARROW_BYTES,
        ),
        (
            4,
            "unknown_dense.model_width",
            vec![GROK1_HIDDEN_DIM, GROK1_HIDDEN_DIM],
            ATTN_MODEL_WIDTH_BYTES,
        ),
        (
            5,
            "unknown_dense.model_width",
            vec![GROK1_HIDDEN_DIM, GROK1_HIDDEN_DIM],
            ATTN_MODEL_WIDTH_BYTES,
        ),
        (
            6,
            "unknown_dense.narrow",
            vec![GROK1_HIDDEN_DIM, 1024],
            ATTN_NARROW_BYTES,
        ),
    ] {
        let policy = if kind.starts_with("moe_expert") {
            "wrap_existing_int8_expert"
        } else {
            "wrap_existing_int8_unknown"
        };
        push_entry(
            entries,
            offset,
            TensorPlan::new(
                &format!("{prefix}.slot_{slot:02}.{kind}"),
                &format!("{prefix}.slot_{slot:02}.{kind}"),
                Some(block),
                Some(slot),
                kind,
                "int8",
                shape,
                bytes,
                policy,
                false,
                None,
            ),
        );
    }

    for slot in 7..=10 {
        push_entry(
            entries,
            offset,
            TensorPlan::new(
                &format!("{prefix}.slot_{slot:02}.block_norm"),
                &format!("{prefix}.slot_{slot:02}.block_norm"),
                Some(block),
                Some(slot),
                "block_norm",
                "f32",
                vec![GROK1_HIDDEN_DIM],
                NORM_BYTES,
                "passthrough_f32_norm",
                protect_norms,
                protect_norms.then_some("block norm protected by policy"),
            ),
        );
    }

    push_entry(
        entries,
        offset,
        TensorPlan::new(
            &format!("{prefix}.slot_11.router"),
            &format!("{prefix}.slot_11.router"),
            Some(block),
            Some(11),
            "router",
            "f32",
            GROK1_ROUTER_SHAPE.to_vec(),
            ROUTER_BYTES,
            "passthrough_f32_router",
            protect_routers,
            protect_routers.then_some("router protected by policy"),
        ),
    );
}

struct TensorPlan<'a> {
    source_tensor_name: &'a str,
    structural_name: &'a str,
    block: Option<usize>,
    slot: Option<usize>,
    kind: &'a str,
    dtype: &'a str,
    shape: Vec<usize>,
    byte_len: u64,
    quant_policy_applied: &'a str,
    protected: bool,
    protected_reason: Option<&'a str>,
}

impl<'a> TensorPlan<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_tensor_name: &'a str,
        structural_name: &'a str,
        block: Option<usize>,
        slot: Option<usize>,
        kind: &'a str,
        dtype: &'a str,
        shape: Vec<usize>,
        byte_len: u64,
        quant_policy_applied: &'a str,
        protected: bool,
        protected_reason: Option<&'a str>,
    ) -> Self {
        Self {
            source_tensor_name,
            structural_name,
            block,
            slot,
            kind,
            dtype,
            shape,
            byte_len,
            quant_policy_applied,
            protected,
            protected_reason,
        }
    }
}

fn push_entry(entries: &mut Vec<ArtifactIndexEntry>, offset: &mut u64, plan: TensorPlan<'_>) {
    let output_checksum = planned_checksum(
        plan.source_tensor_name,
        plan.byte_len,
        plan.quant_policy_applied,
    );
    entries.push(ArtifactIndexEntry {
        source_tensor_name: plan.source_tensor_name.to_string(),
        structural_name: plan.structural_name.to_string(),
        block: plan.block,
        slot: plan.slot,
        kind: plan.kind.to_string(),
        dtype: plan.dtype.to_string(),
        shape: plan.shape,
        byte_len: plan.byte_len,
        source_checksum: None,
        output_checksum,
        quant_policy_applied: plan.quant_policy_applied.to_string(),
        artifact_path: "artifact.saaq-g1-v0.meta".to_string(),
        artifact_offset: *offset,
        protected: plan.protected,
        protected_reason: plan.protected_reason.map(str::to_string),
    });
    *offset += plan.byte_len;
}

fn planned_checksum(name: &str, byte_len: u64, policy: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    hasher.update(byte_len.to_le_bytes());
    hasher.update(policy.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn build_index(
    mode: &str,
    format: &str,
    dry_run: bool,
    entries: Vec<ArtifactIndexEntry>,
) -> ArtifactIndex {
    let source_total_bytes = entries.iter().map(|entry| entry.byte_len).sum();
    let router_count = entries
        .iter()
        .filter(|entry| entry.kind == "router")
        .count();
    ArtifactIndex {
        schema: "grok-ozempic.artifact_index".to_string(),
        schema_version: 1,
        format: format.to_string(),
        model_family: "grok-1".to_string(),
        mode: mode.to_string(),
        dry_run,
        tensor_count: entries.len(),
        source_total_bytes,
        artifact_total_bytes: source_total_bytes,
        router_count,
        entries,
    }
}

fn validate_full_expected_entries(
    entries: &[ArtifactIndexEntry],
    require_router_protection: bool,
    require_norm_protection: bool,
) -> Result<()> {
    if entries.len() != GROK1_TENSOR_TOTAL {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "expected {} tensors before writing final success, got {}",
            GROK1_TENSOR_TOTAL,
            entries.len()
        )));
    }
    let index = build_index("full", GROK1_ARTIFACT_FORMAT, true, entries.to_vec());
    let expected_entries = expected_full_entry_map()?;
    let report = build_validation_report(
        &index,
        GROK1_ARTIFACT_FORMAT,
        require_router_protection,
        require_norm_protection,
        Some(&expected_entries),
        None,
    );
    if report.status != "PASS" {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "planned full artifact failed internal validation: {:?}",
            report.failures
        )));
    }
    Ok(())
}

fn validate_smoke_entries(
    entries: &[ArtifactIndexEntry],
    block: usize,
    include_embedding: bool,
    include_final_norm: bool,
) -> Result<()> {
    let block_entries: Vec<_> = entries
        .iter()
        .filter(|entry| entry.block == Some(block))
        .collect();
    if block_entries.len() != GROK1_BLOCK_TENSORS {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "smoke block_{block:03} tensor count mismatch: expected {}, got {}",
            GROK1_BLOCK_TENSORS,
            block_entries.len()
        )));
    }
    if include_embedding
        && !entries
            .iter()
            .any(|entry| entry.source_tensor_name == "embedding.slot_00.token_embedding")
    {
        return Err(GrokOzempicError::ArtifactValidation(
            "smoke run requested embedding but it is absent from smoke index".to_string(),
        ));
    }
    if include_final_norm
        && !entries
            .iter()
            .any(|entry| entry.source_tensor_name == "final_norm.slot_00.final_norm")
    {
        return Err(GrokOzempicError::ArtifactValidation(
            "smoke run requested final norm but it is absent from smoke index".to_string(),
        ));
    }
    let router_count = block_entries
        .iter()
        .filter(|entry| entry.kind == "router")
        .count();
    if router_count != 1 {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "smoke block_{block:03} router count mismatch: expected 1, got {router_count}"
        )));
    }
    let expert_families = block_entries
        .iter()
        .filter(|entry| entry.kind.starts_with("moe_expert"))
        .count();
    if expert_families != GROK1_EXPERT_FAMILIES_PER_BLOCK {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "smoke block_{block:03} expert family count mismatch: expected {}, got {}",
            GROK1_EXPERT_FAMILIES_PER_BLOCK, expert_families
        )));
    }
    Ok(())
}

fn build_validation_report(
    index: &ArtifactIndex,
    expected_format: &str,
    require_router_protection: bool,
    require_norm_protection: bool,
    expected_entries: Option<&BTreeMap<String, ArtifactIndexEntry>>,
    checksums: Option<&BTreeMap<String, String>>,
) -> ValidationReport {
    let mut failures = Vec::new();
    let warnings = planned_warnings();
    let mut names = HashSet::new();
    let mut duplicates = BTreeSet::new();
    let mut actual_names = BTreeSet::new();
    for entry in &index.entries {
        if !names.insert(entry.source_tensor_name.as_str()) {
            duplicates.insert(entry.source_tensor_name.clone());
        }
        actual_names.insert(entry.source_tensor_name.clone());
    }
    for duplicate in duplicates {
        failures.push(failure(
            "duplicate_tensor",
            Some(duplicate),
            "artifact index contains duplicate tensor entry",
        ));
    }

    if index.schema != "grok-ozempic.artifact_index" {
        failures.push(failure(
            "schema_mismatch",
            None,
            format!(
                "artifact index schema must be grok-ozempic.artifact_index, got {}",
                index.schema
            ),
        ));
    }
    if index.schema_version != 1 {
        failures.push(failure(
            "schema_version_mismatch",
            None,
            format!(
                "artifact index schema_version must be 1, got {}",
                index.schema_version
            ),
        ));
    }
    if index.format != expected_format {
        failures.push(failure(
            "format_mismatch",
            None,
            format!(
                "artifact index format must be {expected_format}, got {}",
                index.format
            ),
        ));
    }
    if index.model_family != "grok-1" {
        failures.push(failure(
            "manifest_artifact_mismatch",
            None,
            "artifact index model_family is not grok-1",
        ));
    }
    if index.mode == "full" && index.entries.len() != GROK1_TENSOR_TOTAL {
        failures.push(failure(
            "missing_tensor",
            None,
            format!(
                "full artifact must contain exactly {GROK1_TENSOR_TOTAL} tensors, got {}",
                index.entries.len()
            ),
        ));
    }
    match index.mode.as_str() {
        "full" | "smoke" => {}
        other => failures.push(failure(
            "mode_mismatch",
            None,
            format!("artifact index mode must be full or smoke, got {other}"),
        )),
    }
    if index.tensor_count != index.entries.len() {
        failures.push(failure(
            "count_mismatch",
            None,
            format!(
                "artifact index tensor_count {} does not match {} entries",
                index.tensor_count,
                index.entries.len()
            ),
        ));
    }
    if let Some(expected_entries) = expected_entries {
        let expected_names: BTreeSet<_> = expected_entries.keys().cloned().collect();
        for missing in expected_names.difference(&actual_names) {
            failures.push(failure(
                "missing_tensor",
                Some(missing.clone()),
                "expected tensor is missing from artifact index",
            ));
        }
        for unexpected in actual_names.difference(&expected_names) {
            failures.push(failure(
                "manifest_artifact_mismatch",
                Some(unexpected.clone()),
                "artifact index tensor is not part of the Grok-1 source inventory",
            ));
        }
        for entry in &index.entries {
            let Some(expected) = expected_entries.get(&entry.source_tensor_name) else {
                continue;
            };
            if entry.block != expected.block
                || entry.slot != expected.slot
                || entry.kind != expected.kind
                || entry.quant_policy_applied != expected.quant_policy_applied
            {
                failures.push(failure(
                    "manifest_artifact_mismatch",
                    Some(entry.source_tensor_name.clone()),
                    format!(
                        "expected block {:?} slot {:?} kind {} policy {}, got block {:?} slot {:?} kind {} policy {}",
                        expected.block,
                        expected.slot,
                        expected.kind,
                        expected.quant_policy_applied,
                        entry.block,
                        entry.slot,
                        entry.kind,
                        entry.quant_policy_applied
                    ),
                ));
            }
            if entry.dtype != expected.dtype {
                failures.push(failure(
                    "dtype_mismatch",
                    Some(entry.source_tensor_name.clone()),
                    format!("expected dtype {}, got {}", expected.dtype, entry.dtype),
                ));
            }
            if entry.shape != expected.shape {
                failures.push(failure(
                    "shape_mismatch",
                    Some(entry.source_tensor_name.clone()),
                    format!("expected shape {:?}, got {:?}", expected.shape, entry.shape),
                ));
            }
            if entry.byte_len != expected.byte_len {
                failures.push(failure(
                    "byte_mismatch",
                    Some(entry.source_tensor_name.clone()),
                    format!(
                        "expected byte length {}, got {}",
                        expected.byte_len, entry.byte_len
                    ),
                ));
            }
            if entry.artifact_path != expected.artifact_path {
                failures.push(failure(
                    "artifact_location_mismatch",
                    Some(entry.source_tensor_name.clone()),
                    format!(
                        "expected artifact_path {}, got {}",
                        expected.artifact_path, entry.artifact_path
                    ),
                ));
            }
            if entry.artifact_offset != expected.artifact_offset {
                failures.push(failure(
                    "artifact_location_mismatch",
                    Some(entry.source_tensor_name.clone()),
                    format!(
                        "expected artifact_offset {}, got {}",
                        expected.artifact_offset, entry.artifact_offset
                    ),
                ));
            }
            if normalize_checksum(&entry.output_checksum)
                != normalize_checksum(&expected.output_checksum)
            {
                failures.push(failure(
                    "checksum_mismatch",
                    Some(entry.source_tensor_name.clone()),
                    format!(
                        "expected output checksum {}, got {}",
                        expected.output_checksum, entry.output_checksum
                    ),
                ));
            }
        }
    }

    let routers: Vec<_> = index
        .entries
        .iter()
        .filter(|entry| entry.kind == "router")
        .collect();
    let expected_routers = if index.mode == "smoke" {
        1
    } else {
        GROK1_BLOCK_COUNT as usize
    };
    if routers.len() != expected_routers {
        failures.push(failure(
            "router_count_mismatch",
            None,
            format!("expected {expected_routers} routers, got {}", routers.len()),
        ));
    }
    if index.router_count != routers.len() {
        failures.push(failure(
            "count_mismatch",
            None,
            format!(
                "artifact index router_count {} does not match {} router entries",
                index.router_count,
                routers.len()
            ),
        ));
    }
    let mut protected_router_violations = 0usize;
    for router in &routers {
        if router.dtype != "f32" {
            failures.push(failure(
                "dtype_mismatch",
                Some(router.source_tensor_name.clone()),
                "router must remain f32",
            ));
        }
        if router.shape != GROK1_ROUTER_SHAPE {
            failures.push(failure(
                "shape_mismatch",
                Some(router.source_tensor_name.clone()),
                "router must remain shape (6144, 8)",
            ));
        }
        if router.quant_policy_applied != "passthrough_f32_router" {
            protected_router_violations += 1;
            failures.push(failure(
                "router_policy_violation",
                Some(router.source_tensor_name.clone()),
                "router must remain pass-through/f32",
            ));
        } else if require_router_protection && !router.protected {
            protected_router_violations += 1;
            failures.push(failure(
                "router_policy_violation",
                Some(router.source_tensor_name.clone()),
                "router must be protected when strict router protection is enabled",
            ));
        }
    }

    let mut protected_norm_violations = 0usize;
    for norm in index
        .entries
        .iter()
        .filter(|entry| entry.kind == "block_norm" || entry.kind == "final_norm")
    {
        if norm.dtype != "f32" {
            failures.push(failure(
                "dtype_mismatch",
                Some(norm.source_tensor_name.clone()),
                "norm tensors must remain f32",
            ));
        }
        if norm.quant_policy_applied != "passthrough_f32_norm" {
            protected_norm_violations += 1;
            failures.push(failure(
                "norm_policy_violation",
                Some(norm.source_tensor_name.clone()),
                "norm tensors must remain pass-through/f32",
            ));
        } else if require_norm_protection && !norm.protected {
            protected_norm_violations += 1;
            failures.push(failure(
                "norm_policy_violation",
                Some(norm.source_tensor_name.clone()),
                "norm tensors must be protected when strict norm protection is enabled",
            ));
        }
    }

    let mut expert_association_count = 0usize;
    let blocks: Vec<_> = if index.mode == "full" {
        (0..GROK1_BLOCK_COUNT as usize).collect()
    } else {
        index
            .entries
            .iter()
            .filter_map(|entry| entry.block)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    };
    for block in blocks {
        let block_entries: Vec<_> = index
            .entries
            .iter()
            .filter(|entry| entry.block == Some(block))
            .collect();
        if block_entries.len() != GROK1_BLOCK_TENSORS {
            failures.push(failure(
                "missing_tensor",
                None,
                format!(
                    "block_{block:03} expected {GROK1_BLOCK_TENSORS} tensors, got {}",
                    block_entries.len()
                ),
            ));
        }
        let families = index
            .entries
            .iter()
            .filter(|entry| entry.block == Some(block) && entry.kind.starts_with("moe_expert"))
            .count();
        if families != GROK1_EXPERT_FAMILIES_PER_BLOCK {
            failures.push(failure(
                "expert_family_missing",
                None,
                format!(
                    "block_{block:03} expected {} expert families, got {families}",
                    GROK1_EXPERT_FAMILIES_PER_BLOCK
                ),
            ));
        }
        let router_count = block_entries
            .iter()
            .filter(|entry| entry.kind == "router")
            .count();
        if router_count != 1 {
            failures.push(failure(
                "router_count_mismatch",
                None,
                format!("block_{block:03} expected 1 router, got {router_count}"),
            ));
        }
        let norm_count = block_entries
            .iter()
            .filter(|entry| entry.kind == "block_norm")
            .count();
        if norm_count != GROK1_NORMS_PER_BLOCK {
            failures.push(failure(
                "missing_tensor",
                None,
                format!(
                    "block_{block:03} expected {GROK1_NORMS_PER_BLOCK} block norms, got {norm_count}"
                ),
            ));
        }
        let unknown_dense_count = block_entries
            .iter()
            .filter(|entry| entry.kind.starts_with("unknown_dense"))
            .count();
        if unknown_dense_count != GROK1_UNKNOWN_DENSE_PER_BLOCK {
            failures.push(failure(
                "missing_tensor",
                None,
                format!(
                    "block_{block:03} expected {GROK1_UNKNOWN_DENSE_PER_BLOCK} unknown dense tensors, got {unknown_dense_count}"
                ),
            ));
        }
        expert_association_count += families * GROK1_EXPERT_COUNT as usize;
    }

    let source_total_bytes: u64 = index.entries.iter().map(|entry| entry.byte_len).sum();
    if index.source_total_bytes != source_total_bytes {
        failures.push(failure(
            "count_mismatch",
            None,
            format!(
                "artifact index source_total_bytes {} does not match recomputed {}",
                index.source_total_bytes, source_total_bytes
            ),
        ));
    }
    if source_total_bytes != index.artifact_total_bytes {
        failures.push(failure(
            "byte_mismatch",
            None,
            format!(
                "source byte accounting {source_total_bytes} != artifact byte accounting {}",
                index.artifact_total_bytes
            ),
        ));
    }
    if index.mode == "full" && source_total_bytes != GROK1_TENSOR_TOTAL_BYTES {
        failures.push(failure(
            "byte_mismatch",
            None,
            format!(
                "full raw source bytes must be {GROK1_TENSOR_TOTAL_BYTES}, got {source_total_bytes}"
            ),
        ));
    }

    let checksum_coverage = if let Some(checksums) = checksums {
        let mut covered = 0usize;
        for entry in &index.entries {
            match checksums.get(&entry.source_tensor_name) {
                Some(expected) => {
                    if normalize_checksum(expected) == normalize_checksum(&entry.output_checksum) {
                        covered += 1;
                    } else {
                        failures.push(failure(
                            "checksum_mismatch",
                            Some(entry.source_tensor_name.clone()),
                            format!(
                                "checksums file expected {}, got {}",
                                expected, entry.output_checksum
                            ),
                        ));
                    }
                }
                None => failures.push(failure(
                    "checksum_missing",
                    Some(entry.source_tensor_name.clone()),
                    "checksums file is missing this tensor entry",
                )),
            }
        }
        for unexpected in checksums
            .keys()
            .filter(|name| !actual_names.contains(*name))
        {
            failures.push(failure(
                "checksum_mismatch",
                Some(unexpected.clone()),
                "checksums file contains a tensor not present in artifact index",
            ));
        }
        format!("{covered}/{} source tensors", index.entries.len())
    } else {
        let covered = index
            .entries
            .iter()
            .filter(|entry| entry.source_checksum.is_some())
            .count();
        format!("{covered}/{} source tensors", index.entries.len())
    };

    ValidationReport {
        status: if failures.is_empty() { "PASS" } else { "FAIL" }.to_string(),
        source_tensor_count: if index.mode == "full" {
            GROK1_TENSOR_TOTAL
        } else {
            index.entries.len()
        },
        artifact_tensor_count: index.entries.len(),
        router_count: routers.len(),
        protected_router_violations,
        protected_norm_violations,
        expert_association_count,
        unknown_unresolved_warning_count: warnings.len(),
        checksum_coverage,
        source_total_bytes,
        artifact_total_bytes: index.artifact_total_bytes,
        byte_accounting_result: if source_total_bytes == index.artifact_total_bytes {
            "match"
        } else {
            "mismatch"
        }
        .to_string(),
        failures,
        warnings,
    }
}

fn failure(category: &str, tensor: Option<String>, message: impl Into<String>) -> FailureRecord {
    FailureRecord {
        category: category.to_string(),
        tensor,
        message: message.into(),
    }
}

fn planned_warnings() -> Vec<WarningRecord> {
    vec![
        WarningRecord {
            category: "unresolved_expert_projection".to_string(),
            tensor: Some("*.slot_00.moe_expert.unresolved".to_string()),
            message: "expert slot 00 is structurally preserved but projection label remains unresolved".to_string(),
        },
        WarningRecord {
            category: "unresolved_expert_projection".to_string(),
            tensor: Some("*.slot_02.moe_expert.unresolved".to_string()),
            message: "expert slot 02 is structurally preserved but projection label remains unresolved".to_string(),
        },
        WarningRecord {
            category: "unknown_dense_slot".to_string(),
            tensor: Some("*.slot_03/04/05/06".to_string()),
            message: "dense attention slots are wrapped as existing int8 payloads unless shape/dtype/count drift occurs".to_string(),
        },
    ]
}

fn write_conversion_outputs(
    output_root: &Path,
    manifest_path: &Path,
    index: &ArtifactIndex,
    warnings: &[WarningRecord],
    dry_run: bool,
) -> Result<()> {
    fs::create_dir_all(output_root)?;
    fs::copy(manifest_path, output_root.join("manifest.used.json"))?;
    write_json(output_root.join("artifact.index.json"), index)?;
    write_json(output_root.join("checksums.json"), &output_checksums(index))?;
    write_json(output_root.join("warnings.json"), warnings)?;
    if !dry_run {
        fs::write(
            output_root.join("artifact.saaq-g1-v0.meta"),
            artifact_meta(index),
        )?;
    }
    fs::write(
        output_root.join("conversion.summary.md"),
        conversion_summary(index),
    )?;
    Ok(())
}

fn write_smoke_outputs(
    output_root: &Path,
    index: &ArtifactIndex,
    warnings: &[WarningRecord],
    dry_run: bool,
) -> Result<()> {
    fs::create_dir_all(output_root)?;
    write_json(output_root.join("smoke.index.json"), index)?;
    write_json(
        output_root.join("smoke.checksums.json"),
        &output_checksums(index),
    )?;
    write_json(output_root.join("smoke.warnings.json"), warnings)?;
    if !dry_run {
        fs::write(
            output_root.join("artifact.saaq-g1-v0.meta"),
            artifact_meta(index),
        )?;
    }
    fs::write(output_root.join("smoke.summary.md"), smoke_summary(index))?;
    Ok(())
}

fn write_validation_outputs(output_root: &Path, report: &ValidationReport) -> Result<()> {
    fs::create_dir_all(output_root)?;
    fs::write(
        output_root.join("validation.summary.md"),
        validation_summary(report),
    )?;
    write_json(output_root.join("validation.report.json"), report)?;
    write_json(
        output_root.join("validation.failures.json"),
        &report.failures,
    )?;
    write_json(
        output_root.join("validation.warnings.json"),
        &report.warnings,
    )?;
    Ok(())
}

fn output_checksums(index: &ArtifactIndex) -> BTreeMap<String, String> {
    index
        .entries
        .iter()
        .map(|entry| {
            (
                entry.source_tensor_name.clone(),
                entry.output_checksum.clone(),
            )
        })
        .collect()
}

fn expected_full_entry_map() -> Result<BTreeMap<String, ArtifactIndexEntry>> {
    Ok(planned_grok1_entries(false, false, None)?
        .into_iter()
        .map(|entry| (entry.source_tensor_name.clone(), entry))
        .collect())
}

fn write_json<T: Serialize + ?Sized>(path: PathBuf, value: &T) -> Result<()> {
    let body = serde_json::to_string_pretty(value).map_err(|e| {
        GrokOzempicError::ArtifactValidation(format!("failed to serialize {}: {e}", path.display()))
    })?;
    fs::write(path, format!("{body}\n"))?;
    Ok(())
}

fn artifact_meta(index: &ArtifactIndex) -> String {
    let mut body = String::from("# grok-ozempic metadata-only artifact\n");
    for entry in &index.entries {
        body.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            entry.artifact_offset, entry.byte_len, entry.dtype, entry.source_tensor_name
        ));
    }
    body
}

fn conversion_summary(index: &ArtifactIndex) -> String {
    let router_total = index
        .entries
        .iter()
        .filter(|entry| entry.kind == "router")
        .count();
    let protected_routers = index
        .entries
        .iter()
        .filter(|entry| entry.kind == "router" && entry.protected)
        .count();
    let norm_total = index
        .entries
        .iter()
        .filter(|entry| entry.kind == "block_norm" || entry.kind == "final_norm")
        .count();
    let protected_norms = index
        .entries
        .iter()
        .filter(|entry| {
            (entry.kind == "block_norm" || entry.kind == "final_norm") && entry.protected
        })
        .count();
    format!(
        "# Grok-1 conversion summary\n\n- status: PASS\n- mode: {}\n- dry_run: {}\n- format: {}\n- tensor_count: {}\n- router_count: {}\n- source_total_bytes: {}\n- artifact_total_bytes: {}\n- f32_tensors: {}\n- int8_tensors: {}\n- protected routers: {protected_routers}/{router_total}\n- protected norms: {protected_norms}/{norm_total}\n- int8 wrapping policy: expert and unknown tensors are wrapped with metadata.\n",
        index.mode,
        index.dry_run,
        index.format,
        index.tensor_count,
        index.router_count,
        index.source_total_bytes,
        index.artifact_total_bytes,
        index
            .entries
            .iter()
            .filter(|entry| entry.dtype == "f32")
            .count(),
        index
            .entries
            .iter()
            .filter(|entry| entry.dtype == "int8")
            .count(),
    )
}

fn smoke_summary(index: &ArtifactIndex) -> String {
    let block = index
        .entries
        .iter()
        .find_map(|entry| entry.block)
        .unwrap_or(0);
    format!(
        "# Grok-1 smoke summary\n\n- status: PASS\n- block: block_{block:03}\n- dry_run: {}\n- tensor_count: {}\n- block_tensor_count: {}\n- router_count: {}\n- expert_families: {}\n- warnings: {}\n",
        index.dry_run,
        index.tensor_count,
        index
            .entries
            .iter()
            .filter(|entry| entry.block == Some(block))
            .count(),
        index.router_count,
        index
            .entries
            .iter()
            .filter(|entry| entry.kind.starts_with("moe_expert"))
            .count(),
        planned_warnings().len(),
    )
}

fn validation_summary(report: &ValidationReport) -> String {
    format!(
        "# Grok-1 artifact validation summary\n\n- status: {}\n- source tensor count: {}\n- artifact tensor count: {}\n- router count: {}\n- protected router violations: {}\n- protected norm violations: {}\n- expert association count: {}\n- unknown/unresolved warning count: {}\n- checksum coverage: {}\n- byte accounting result: {}\n- source total bytes: {}\n- artifact total bytes: {}\n- failure count: {}\n",
        report.status,
        report.source_tensor_count,
        report.artifact_tensor_count,
        report.router_count,
        report.protected_router_violations,
        report.protected_norm_violations,
        report.expert_association_count,
        report.unknown_unresolved_warning_count,
        report.checksum_coverage,
        report.byte_accounting_result,
        report.source_total_bytes,
        report.artifact_total_bytes,
        report.failures.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::manifest::GROK1_BASELINE_JSON;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("grok_ozempic_{name}_{stamp}"));
        fs::create_dir_all(&path).expect("temp dir");
        path
    }

    fn write_manifest(dir: &Path) -> PathBuf {
        let path = dir.join("manifest.json");
        fs::write(&path, GROK1_BASELINE_JSON).expect("manifest write");
        path
    }

    #[test]
    fn ingest_rejects_version_mismatch() {
        let dir = temp_dir("bad_version");
        let path = dir.join("manifest.json");
        fs::write(
            &path,
            GROK1_BASELINE_JSON.replace("\"schema_version\": 1", "\"schema_version\": 99"),
        )
        .expect("manifest write");
        let err = validate_ingest_path(&path, None).expect_err("bad version must fail");
        assert!(err.to_string().contains("schema_version"));
    }

    #[test]
    fn convert_writes_deterministic_full_index() {
        let dir = temp_dir("convert");
        let manifest = write_manifest(&dir);
        let out = dir.join("out");
        let index = convert_grok1(ConvertOptions {
            checkpoint: None,
            manifest: &manifest,
            output_root: &out,
            format: GROK1_ARTIFACT_FORMAT,
            protect_routers: true,
            protect_norms: true,
            dry_run: true,
        })
        .expect("convert");
        assert_eq!(index.tensor_count, GROK1_TENSOR_TOTAL);
        assert_eq!(index.router_count, GROK1_BLOCK_COUNT as usize);
        assert_eq!(index.source_total_bytes, GROK1_TENSOR_TOTAL_BYTES);
        assert!(out.join("artifact.index.json").is_file());
        assert!(out.join("conversion.summary.md").is_file());
        assert!(out.join("checksums.json").is_file());
        assert!(out.join("warnings.json").is_file());
        assert!(!out.join("artifact.saaq-g1-v0.meta").exists());
    }

    #[test]
    fn smoke_selects_one_block_with_embedding_and_final_norm() {
        let dir = temp_dir("smoke");
        let manifest = write_manifest(&dir);
        let out = dir.join("smoke-out");
        let index = smoke_grok1(SmokeOptions {
            checkpoint: None,
            manifest: &manifest,
            block: 0,
            include_embedding: true,
            include_final_norm: true,
            output_root: &out,
            dry_run: true,
        })
        .expect("smoke");
        assert_eq!(index.tensor_count, GROK1_BLOCK_TENSORS + 2);
        assert_eq!(index.router_count, 1);
        assert_eq!(
            index
                .entries
                .iter()
                .filter(|entry| entry.block == Some(0))
                .count(),
            GROK1_BLOCK_TENSORS
        );
        assert!(out.join("smoke.index.json").is_file());
        assert!(out.join("smoke.summary.md").is_file());
    }

    #[test]
    fn convert_allows_unprotected_routers_and_norms() {
        let dir = temp_dir("convert_unprotected");
        let manifest = write_manifest(&dir);
        let out = dir.join("out");
        let index = convert_grok1(ConvertOptions {
            checkpoint: None,
            manifest: &manifest,
            output_root: &out,
            format: GROK1_ARTIFACT_FORMAT,
            protect_routers: false,
            protect_norms: false,
            dry_run: true,
        })
        .expect("convert");
        assert!(
            index
                .entries
                .iter()
                .filter(|entry| entry.kind == "router")
                .all(|entry| !entry.protected)
        );
        assert!(
            index
                .entries
                .iter()
                .filter(|entry| entry.kind == "block_norm" || entry.kind == "final_norm")
                .all(|entry| !entry.protected)
        );
        let summary = fs::read_to_string(out.join("conversion.summary.md")).expect("summary");
        assert!(summary.contains("- protected routers: 0/64"));
        assert!(summary.contains("- protected norms: 0/257"));
    }

    #[test]
    fn checkpoint_checksums_reject_escape_paths() {
        let dir = temp_dir("checkpoint_escape");
        let checkpoint = dir.join("ckpt");
        fs::create_dir_all(&checkpoint).expect("checkpoint dir");
        fs::write(
            checkpoint.join("checksums.json"),
            serde_json::to_vec_pretty(&json!({
                "../escape.bin": "sha256:deadbeef",
            }))
            .expect("checksums json"),
        )
        .expect("checksums write");
        let err = validate_checkpoint_checksums(&checkpoint).expect_err("escape must fail");
        assert!(
            err.to_string()
                .contains("illegal path traversal or absolute path")
        );
    }

    #[test]
    fn checkpoint_checksums_accept_uppercase_sha_prefix() {
        let dir = temp_dir("checkpoint_sha_prefix");
        let checkpoint = dir.join("ckpt");
        fs::create_dir_all(&checkpoint).expect("checkpoint dir");
        let payload = checkpoint.join("payload.bin");
        fs::write(&payload, b"grok-ozempic").expect("payload write");
        let digest = sha256_file(&payload).expect("digest");
        fs::write(
            checkpoint.join("checksums.json"),
            serde_json::to_vec_pretty(&json!({
                "payload.bin": format!("SHA256:{digest}"),
            }))
            .expect("checksums json"),
        )
        .expect("checksums write");
        validate_checkpoint_checksums(&checkpoint).expect("uppercase prefix should normalize");
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_checksums_reject_symlink_escape() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir("checkpoint_symlink_escape");
        let checkpoint = dir.join("ckpt");
        let outside = dir.join("outside");
        fs::create_dir_all(&checkpoint).expect("checkpoint dir");
        fs::create_dir_all(&outside).expect("outside dir");
        let target = outside.join("payload.bin");
        fs::write(&target, b"grok-ozempic").expect("payload write");
        symlink(&target, checkpoint.join("payload-link.bin")).expect("symlink");
        let digest = sha256_file(&target).expect("digest");
        fs::write(
            checkpoint.join("checksums.json"),
            serde_json::to_vec_pretty(&json!({
                "payload-link.bin": format!("sha256:{digest}"),
            }))
            .expect("checksums json"),
        )
        .expect("checksums write");
        let err = validate_checkpoint_checksums(&checkpoint).expect_err("symlink escape must fail");
        assert!(err.to_string().contains("resolves outside checkpoint root"));
    }

    #[test]
    fn validator_reports_negative_cases() {
        let dir = temp_dir("validate_neg");
        let manifest = write_manifest(&dir);
        let out = dir.join("out");
        let mut index = convert_grok1(ConvertOptions {
            checkpoint: None,
            manifest: &manifest,
            output_root: &out,
            format: GROK1_ARTIFACT_FORMAT,
            protect_routers: true,
            protect_norms: true,
            dry_run: true,
        })
        .expect("convert");
        index
            .entries
            .retain(|entry| entry.source_tensor_name != "block_000.slot_11.router");
        if let Some(router) = index
            .entries
            .iter_mut()
            .find(|entry| entry.kind == "router")
        {
            router.dtype = "int8".to_string();
        }
        index.entries.push(index.entries[0].clone());
        if let Some(expert) = index
            .entries
            .iter_mut()
            .find(|entry| entry.block == Some(1) && entry.kind.starts_with("moe_expert"))
        {
            expert.kind = "removed_expert".to_string();
        }
        index.artifact_total_bytes += 1;
        let expected_entries = expected_full_entry_map().expect("expected entries");
        let report = build_validation_report(
            &index,
            GROK1_ARTIFACT_FORMAT,
            true,
            true,
            Some(&expected_entries),
            None,
        );
        for category in [
            "router_count_mismatch",
            "dtype_mismatch",
            "duplicate_tensor",
            "expert_family_missing",
            "byte_mismatch",
        ] {
            assert!(
                report
                    .failures
                    .iter()
                    .any(|failure| failure.category == category),
                "missing failure category {category}: {:?}",
                report.failures
            );
        }
    }

    #[test]
    fn validator_enforces_format_inventory_and_supplied_checksums() {
        let dir = temp_dir("validate_contract");
        let manifest = write_manifest(&dir);
        let out = dir.join("out");
        let mut index = convert_grok1(ConvertOptions {
            checkpoint: None,
            manifest: &manifest,
            output_root: &out,
            format: GROK1_ARTIFACT_FORMAT,
            protect_routers: true,
            protect_norms: true,
            dry_run: true,
        })
        .expect("convert");
        let checksums = output_checksums(&index);
        index.format = "custom-grok1".to_string();
        index.schema = "foreign.artifact_index".to_string();
        index.schema_version = 9;
        index.mode = "surprise".to_string();
        index.tensor_count -= 1;
        index.router_count -= 1;
        index.source_total_bytes -= 1;
        if let Some(router) = index
            .entries
            .iter_mut()
            .find(|entry| entry.source_tensor_name == "block_000.slot_11.router")
        {
            router.block = Some(1);
        }
        if let Some(entry) = index
            .entries
            .iter_mut()
            .find(|entry| entry.source_tensor_name == "block_001.slot_00.moe_expert.unresolved")
        {
            entry.artifact_path = "tampered.meta".to_string();
            entry.artifact_offset += 123;
        }
        if let Some(final_norm) = index
            .entries
            .iter_mut()
            .find(|entry| entry.source_tensor_name == "final_norm.slot_00.final_norm")
        {
            final_norm.source_tensor_name = "final_norm.slot_00.renamed".to_string();
        }
        if let Some(entry) = index.entries.first_mut() {
            entry.output_checksum = "sha256:deadbeef".to_string();
        }
        let expected_entries = expected_full_entry_map().expect("expected entries");
        let report = build_validation_report(
            &index,
            GROK1_ARTIFACT_FORMAT,
            true,
            true,
            Some(&expected_entries),
            Some(&checksums),
        );
        for category in [
            "schema_mismatch",
            "schema_version_mismatch",
            "format_mismatch",
            "mode_mismatch",
            "count_mismatch",
            "missing_tensor",
            "manifest_artifact_mismatch",
            "artifact_location_mismatch",
            "checksum_mismatch",
            "checksum_missing",
            "router_count_mismatch",
        ] {
            assert!(
                report
                    .failures
                    .iter()
                    .any(|failure| failure.category == category),
                "missing failure category {category}: {:?}",
                report.failures
            );
        }
    }

    #[test]
    fn validate_artifact_without_output_root_returns_inline_hint() {
        let dir = temp_dir("validate_inline_hint");
        let manifest = write_manifest(&dir);
        let out = dir.join("out");
        let mut index = convert_grok1(ConvertOptions {
            checkpoint: None,
            manifest: &manifest,
            output_root: &out,
            format: GROK1_ARTIFACT_FORMAT,
            protect_routers: true,
            protect_norms: true,
            dry_run: true,
        })
        .expect("convert");
        index.format = "bad-format".to_string();
        let artifact_index = dir.join("artifact.index.json");
        write_json(artifact_index.clone(), &index).expect("artifact index write");
        let err = validate_grok1_artifact(&manifest, &artifact_index, None, None, true)
            .expect_err("validation should fail");
        let message = err.to_string();
        assert!(message.contains("first failure ["));
        assert!(message.contains("rerun with --output-root"));
    }
}
