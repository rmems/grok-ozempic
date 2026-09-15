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

    let mut tensors = Vec::with_capacity(doc.tensors.len());
    for (i, t) in doc.tensors.iter().enumerate() {
        let numel = numel(&t.shape).ok_or_else(|| {
            GrokOzempicError::ArtifactValidation(format!(
                "{label}: tensors[{i}] shape {:?} overflows u64 element count",
                t.shape
            ))
        })?;
        let expected_nbytes = numel.checked_mul(t.dtype.itemsize()).ok_or_else(|| {
            GrokOzempicError::ArtifactValidation(format!(
                "{label}: tensors[{i}] nbytes overflows u64"
            ))
        })?;
        if t.nbytes != expected_nbytes {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "{label}: tensors[{i}] nbytes {} does not match shape {:?} dtype {:?} (expected {expected_nbytes})",
                t.nbytes, t.shape, t.dtype
            )));
        }
        tensors.push(ScanTensor {
            dtype: t.dtype,
            shape: t.shape.clone(),
            nbytes: t.nbytes,
            role: t.role,
        });
    }

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
        let n = numel(&t.shape).ok_or_else(|| {
            GrokOzempicError::ArtifactValidation(
                "inventory scan element count overflows u64".to_string(),
            )
        })?;
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
    let tensor_count: u64 = blocks.iter().map(|b| u64::from(b.tensor_count)).sum();
    let nbytes: u64 = blocks.iter().map(|b| b.total_nbytes).sum();
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

#[cfg(test)]
pub(crate) fn grok1_spec_inventory_scan() -> InventoryScan {
    use crate::core::stream::GROK1_BLOCK_COUNT;
    use crate::types::{
        GROK1_BLOCK_SLOTS, GROK1_HIDDEN_DIM, GROK1_TENSOR_TOTAL_BYTES, GROK1_VOCAB_SIZE,
    };

    let mut tensors = Vec::with_capacity(770);
    tensors.push(ScanTensor {
        dtype: ScanDtype::F32,
        shape: vec![GROK1_VOCAB_SIZE as u64, GROK1_HIDDEN_DIM as u64],
        nbytes: (GROK1_VOCAB_SIZE * GROK1_HIDDEN_DIM * 4) as u64,
        role: ScanRole::Tensor,
    });
    for _ in 0..GROK1_BLOCK_COUNT {
        for slot in GROK1_BLOCK_SLOTS.iter() {
            let dtype = if slot.is_int8 {
                ScanDtype::I8
            } else {
                ScanDtype::F32
            };
            let role = if slot.is_int8 {
                ScanRole::QuantWeight
            } else {
                ScanRole::Tensor
            };
            tensors.push(ScanTensor {
                dtype,
                shape: slot.shape.iter().map(|d| *d as u64).collect(),
                nbytes: slot.bytes,
                role,
            });
        }
    }
    tensors.push(ScanTensor {
        dtype: ScanDtype::F32,
        shape: vec![GROK1_HIDDEN_DIM as u64],
        nbytes: (GROK1_HIDDEN_DIM * 4) as u64,
        role: ScanRole::Tensor,
    });

    let totals = totals_from_tensors(&tensors).expect("spec scan totals");
    assert_eq!(totals.total_bytes, GROK1_TENSOR_TOTAL_BYTES);
    InventoryScan {
        model_family: GROK1_FAMILY.to_string(),
        checkpoint_path: "grok-1-official/ckpt-0".to_string(),
        shard_count: tensors.len(),
        tensors,
        totals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        GROK1_TENSOR_F32, GROK1_TENSOR_INT8, GROK1_TENSOR_QUANT, GROK1_TENSOR_TOTAL,
        GROK1_TENSOR_TOTAL_BYTES, GROK1_TENSOR_TOTAL_ELEMENTS,
    };

    fn mini_inventory_json(totals_override: Option<&str>) -> String {
        let totals = totals_override.unwrap_or(
            r#"{
                "tensors": 2,
                "quant_tensors": 1,
                "f32_tensors": 1,
                "i8_tensors": 1,
                "total_nbytes": 20,
                "total_elements": 9
            }"#,
        );
        format!(
            r#"{{
                "model_family": "grok-1",
                "checkpoint_path": "/fixtures/ckpt-0",
                "shard_count": 2,
                "inferred": {{ "d_model": 4 }},
                "tensors": [
                    {{
                        "shard_path": "/fixtures/t0",
                        "shard_ordinal": 0,
                        "in_shard_index": 0,
                        "role": "tensor",
                        "dtype": "f32",
                        "shape": [4],
                        "offset": 0,
                        "nbytes": 16,
                        "kind": {{ "kind": "block_norm" }},
                        "block_index": null,
                        "block_slot": null
                    }},
                    {{
                        "shard_path": "/fixtures/t1",
                        "shard_ordinal": 1,
                        "in_shard_index": 0,
                        "role": "quant_weight",
                        "dtype": "i8",
                        "shape": [5],
                        "offset": 0,
                        "nbytes": 5,
                        "kind": {{ "kind": "moe_expert_projection", "detail": {{ "projection": "gate" }} }},
                        "block_index": 0,
                        "block_slot": 0
                    }}
                ],
                "blocks": [
                    {{
                        "block_index": null,
                        "label": "embedding",
                        "shard_range": {{ "start": 0, "end_inclusive": 0 }},
                        "tensor_count": 1,
                        "total_nbytes": 16,
                        "dtypes": ["f32"],
                        "kinds": [{{ "kind_label": "block_norm", "count": 1, "nbytes": 16 }}]
                    }},
                    {{
                        "block_index": 0,
                        "label": "block_000",
                        "shard_range": {{ "start": 1, "end_inclusive": 1 }},
                        "tensor_count": 1,
                        "total_nbytes": 5,
                        "dtypes": ["i8"],
                        "kinds": [{{ "kind_label": "moe_expert.gate", "count": 1, "nbytes": 5 }}]
                    }}
                ],
                "totals": {totals},
                "schema_version": 2
            }}"#
        )
    }

    #[test]
    fn parses_inventory_json_and_derives_totals_from_tensors() {
        let scan = parse_inventory_scan_bytes(mini_inventory_json(None).as_bytes(), "<mini>")
            .expect("mini inventory should parse");
        assert_eq!(scan.totals.total, 2);
        assert_eq!(scan.totals.f32_tensors, 1);
        assert_eq!(scan.totals.int8_tensors, 1);
        assert_eq!(scan.totals.quant_tensors, 1);
        assert_eq!(scan.totals.total_elements, 9);
        assert_eq!(scan.totals.total_bytes, 20);
        assert_eq!(scan.shard_count, 2);
    }

    #[test]
    fn rejects_declared_totals_that_do_not_match_tensors() {
        let json = mini_inventory_json(Some(
            r#"{
                "tensors": 99,
                "quant_tensors": 1,
                "f32_tensors": 1,
                "i8_tensors": 1,
                "total_nbytes": 20,
                "total_elements": 9
            }"#,
        ));
        let err = parse_inventory_scan_bytes(json.as_bytes(), "<lie>").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("do not match the `tensors` array"),
            "got {msg}"
        );
        assert!(msg.contains("tensors: declared 99, derived 2"), "got {msg}");
    }

    #[test]
    fn rejects_nbytes_inconsistent_with_shape_and_dtype() {
        let json = mini_inventory_json(None).replace("\"nbytes\": 16", "\"nbytes\": 15");
        let err = parse_inventory_scan_bytes(json.as_bytes(), "<bad-nbytes>").unwrap_err();
        assert!(
            err.to_string().contains("nbytes 15 does not match"),
            "got {err}"
        );
    }

    #[test]
    fn rejects_unsupported_schema_version() {
        let json =
            mini_inventory_json(None).replace("\"schema_version\": 2", "\"schema_version\": 1");
        let err = parse_inventory_scan_bytes(json.as_bytes(), "<v1>").unwrap_err();
        assert!(
            matches!(
                err,
                GrokOzempicError::ManifestSchemaVersion {
                    got: 1,
                    expected: 2
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn grok1_spec_scan_matches_crate_constants() {
        let scan = grok1_spec_inventory_scan();
        assert_eq!(scan.totals.total, GROK1_TENSOR_TOTAL);
        assert_eq!(scan.totals.f32_tensors, GROK1_TENSOR_F32);
        assert_eq!(scan.totals.int8_tensors, GROK1_TENSOR_INT8);
        assert_eq!(scan.totals.quant_tensors, GROK1_TENSOR_QUANT);
        assert_eq!(scan.totals.total_elements, GROK1_TENSOR_TOTAL_ELEMENTS);
        assert_eq!(scan.totals.total_bytes, GROK1_TENSOR_TOTAL_BYTES);
    }
}
