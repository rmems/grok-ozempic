//! Consume an xai-dissect **inventory.json** scan (schema v2).
//!
//! The policy manifest (`xai-dissect.manifest`) has no per-tensor dtype or
//! byte counts — [`crate::core::manifest::ManifestBlock`] is advisory identity
//! only. The real scan lives in a different document: `exports/<slug>/inventory.json`,
//! the canonical tensor catalog in xai-dissect's grok-ozempic handoff contract.
//! This module reads that existing schema. It does **not** invent policy-manifest
//! fields.
//!
//! Derived totals are computed from the `tensors` array. Declared `totals`
//! (and, when present, per-block summaries) must match that sum or the load
//! hard-errors — a document that disagrees with itself is rejected rather than
//! ignored.

use crate::error::{GrokOzempicError, Result};
use crate::reports::schema::TensorTotals;
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Inventory schema version understood by this loader (`ModelInventory`).
pub const INVENTORY_SCAN_SCHEMA_VERSION: u32 = 2;

const GROK1_FAMILY: &str = "grok-1";

/// Parsed xai-dissect `inventory.json` after self-consistency checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InventoryScan {
    pub model_family: String,
    pub checkpoint_path: String,
    pub shard_count: usize,
    pub tensors: Vec<ScanTensor>,
    /// Totals **derived from `tensors`**, not copied from the declared object.
    pub totals: TensorTotals,
}

/// One tensor from the inventory scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanTensor {
    pub dtype: ScanDtype,
    pub shape: Vec<u64>,
    pub nbytes: u64,
    pub role: ScanRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanDtype {
    F32,
    I8,
}

impl ScanDtype {
    pub fn itemsize(self) -> u64 {
        match self {
            ScanDtype::F32 => 4,
            ScanDtype::I8 => 1,
        }
    }
}

/// Parser-level role as spelled in inventory.json (`quant_weight`, not the
/// human label `quant.weight`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanRole {
    Tensor,
    QuantWeight,
    QuantScales,
}

impl ScanRole {
    fn is_quant(self) -> bool {
        matches!(self, ScanRole::QuantWeight | ScanRole::QuantScales)
    }
}

#[derive(Debug, Deserialize)]
struct InventoryScanDoc {
    #[serde(default)]
    model_family: String,
    #[serde(default)]
    checkpoint_path: String,
    shard_count: u32,
    tensors: Vec<ScanTensorDoc>,
    #[serde(default)]
    blocks: Vec<BlockSummaryDoc>,
    totals: DeclaredTotals,
    schema_version: u32,
}

#[derive(Debug, Deserialize)]
struct ScanTensorDoc {
    dtype: ScanDtype,
    shape: Vec<u64>,
    nbytes: u64,
    role: ScanRole,
}

#[derive(Debug, Deserialize)]
struct BlockSummaryDoc {
    tensor_count: u32,
    total_nbytes: u64,
}

#[derive(Debug, Deserialize)]
struct DeclaredTotals {
    tensors: u64,
    quant_tensors: u64,
    f32_tensors: u64,
    i8_tensors: u64,
    total_nbytes: u64,
    total_elements: u64,
}

/// Load and validate an xai-dissect `inventory.json` from disk.
pub fn load_inventory_scan(path: &Path) -> Result<InventoryScan> {
    let bytes = fs::read(path).map_err(|e| GrokOzempicError::ManifestIo {
        path: path.display().to_string(),
        source: e,
    })?;
    parse_inventory_scan_bytes(&bytes, &path.display().to_string())
}

/// Parse already-loaded inventory.json bytes.
///
/// Unknown top-level fields are tolerated. The document must be internally
/// consistent: declared `totals` equal the sum of `tensors`, each tensor's
/// `nbytes` equals `numel * itemsize`, and block summaries (when present)
/// add up to the same tensor/byte totals.
pub fn parse_inventory_scan_bytes(bytes: &[u8], label: &str) -> Result<InventoryScan> {
    let doc: InventoryScanDoc =
        serde_json::from_slice(bytes).map_err(|e| GrokOzempicError::ManifestParse {
            path: label.to_string(),
            source: e,
        })?;
    validate_scan_header(&doc, label)?;

    let tensors = doc
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| scan_tensor_from_doc(t, i, label))
        .collect::<Result<Vec<_>>>()?;

    let derived = totals_from_tensors(&tensors)?;
    reject_totals_mismatch(label, &doc.totals, &derived)?;
    reject_block_mismatch(label, &doc.blocks, &derived)?;

    Ok(InventoryScan {
        model_family: doc.model_family,
        checkpoint_path: doc.checkpoint_path,
        shard_count: doc.shard_count as usize,
        tensors,
        totals: derived,
    })
}

fn validate_scan_header(doc: &InventoryScanDoc, label: &str) -> Result<()> {
    if doc.schema_version != INVENTORY_SCAN_SCHEMA_VERSION {
        return Err(GrokOzempicError::ManifestSchemaVersion {
            got: doc.schema_version,
            expected: INVENTORY_SCAN_SCHEMA_VERSION,
        });
    }
    if doc.model_family != GROK1_FAMILY {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "{label}: inventory scan currently supports only {GROK1_FAMILY}; got {}",
            doc.model_family
        )));
    }
    if doc.tensors.is_empty() {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "{label}: inventory scan `tensors` array is empty"
        )));
    }
    if doc.shard_count == 0 {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "{label}: inventory scan shard_count is 0 but tensors is nonempty"
        )));
    }
    Ok(())
}

fn scan_tensor_from_doc(t: &ScanTensorDoc, index: usize, label: &str) -> Result<ScanTensor> {
    let numel = numel(&t.shape).ok_or_else(|| {
        GrokOzempicError::ArtifactValidation(format!(
            "{label}: tensors[{index}] shape {:?} overflows u64 element count",
            t.shape
        ))
    })?;
    let expected_nbytes = numel.checked_mul(t.dtype.itemsize()).ok_or_else(|| {
        GrokOzempicError::ArtifactValidation(format!(
            "{label}: tensors[{index}] nbytes overflows u64"
        ))
    })?;
    if t.nbytes != expected_nbytes {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "{label}: tensors[{index}] nbytes {} does not match shape {:?} dtype {:?} (expected {expected_nbytes})",
            t.nbytes, t.shape, t.dtype
        )));
    }
    Ok(ScanTensor {
        dtype: t.dtype,
        shape: t.shape.clone(),
        nbytes: t.nbytes,
        role: t.role,
    })
}

fn numel(shape: &[u64]) -> Option<u64> {
    shape.iter().try_fold(1u64, |acc, d| acc.checked_mul(*d))
}

fn totals_from_tensors(tensors: &[ScanTensor]) -> Result<TensorTotals> {
    let mut f32_tensors = 0usize;
    let mut int8_tensors = 0usize;
    let mut quant_tensors = 0usize;
    let mut total_elements = 0u64;
    let mut total_bytes = 0u64;

    for t in tensors {
        match t.dtype {
            ScanDtype::F32 => f32_tensors += 1,
            ScanDtype::I8 => int8_tensors += 1,
        }
        if t.role.is_quant() {
            quant_tensors += 1;
        }
        // `scan_tensor_from_doc` already proved nbytes == numel * itemsize.
        let n = t.nbytes / t.dtype.itemsize();
        total_elements = total_elements.checked_add(n).ok_or_else(|| {
            GrokOzempicError::ArtifactValidation(
                "inventory scan total_elements overflows u64".to_string(),
            )
        })?;
        total_bytes = total_bytes.checked_add(t.nbytes).ok_or_else(|| {
            GrokOzempicError::ArtifactValidation(
                "inventory scan total_bytes overflows u64".to_string(),
            )
        })?;
    }

    Ok(TensorTotals {
        total: tensors.len(),
        f32_tensors,
        int8_tensors,
        quant_tensors,
        total_elements,
        total_bytes,
    })
}

fn reject_totals_mismatch(
    label: &str,
    declared: &DeclaredTotals,
    derived: &TensorTotals,
) -> Result<()> {
    let mut mismatches = Vec::new();
    push_mismatch(
        &mut mismatches,
        "tensors",
        declared.tensors,
        derived.total as u64,
    );
    push_mismatch(
        &mut mismatches,
        "f32_tensors",
        declared.f32_tensors,
        derived.f32_tensors as u64,
    );
    push_mismatch(
        &mut mismatches,
        "i8_tensors",
        declared.i8_tensors,
        derived.int8_tensors as u64,
    );
    push_mismatch(
        &mut mismatches,
        "quant_tensors",
        declared.quant_tensors,
        derived.quant_tensors as u64,
    );
    push_mismatch(
        &mut mismatches,
        "total_elements",
        declared.total_elements,
        derived.total_elements,
    );
    push_mismatch(
        &mut mismatches,
        "total_nbytes",
        declared.total_nbytes,
        derived.total_bytes,
    );
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(GrokOzempicError::ArtifactValidation(format!(
        "{label}: inventory scan `totals` do not match the `tensors` array ({})",
        mismatches.join("; ")
    )))
}

fn push_mismatch(out: &mut Vec<String>, field: &str, declared: u64, derived: u64) {
    if declared != derived {
        out.push(format!("{field}: declared {declared}, derived {derived}"));
    }
}

fn reject_block_mismatch(
    label: &str,
    blocks: &[BlockSummaryDoc],
    derived: &TensorTotals,
) -> Result<()> {
    if blocks.is_empty() {
        return Ok(());
    }
    let mut tensor_count = 0u64;
    let mut nbytes = 0u64;
    for block in blocks {
        tensor_count = tensor_count
            .checked_add(u64::from(block.tensor_count))
            .ok_or_else(|| {
                GrokOzempicError::ArtifactValidation(format!(
                    "{label}: inventory scan block tensor_count overflows u64"
                ))
            })?;
        nbytes = nbytes.checked_add(block.total_nbytes).ok_or_else(|| {
            GrokOzempicError::ArtifactValidation(format!(
                "{label}: inventory scan block total_nbytes overflows u64"
            ))
        })?;
    }
    if tensor_count != derived.total as u64 {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "{label}: inventory scan block summaries count {tensor_count} tensors, tensors array has {}",
            derived.total
        )));
    }
    if nbytes != derived.total_bytes {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "{label}: inventory scan block summaries sum to {nbytes} bytes, tensors array sums to {}",
            derived.total_bytes
        )));
    }
    Ok(())
}
