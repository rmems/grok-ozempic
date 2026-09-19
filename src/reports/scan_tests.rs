//! Inventory-scan loader tests. Kept out of `scan.rs` so cyclomatic-complexity
//! engines do not treat fixtures as production functions.

use super::scan::{
    InventoryScan, ScanDtype, ScanRole, load_inventory_scan, parse_inventory_scan_bytes,
};
use crate::core::stream::GROK1_BLOCK_COUNT;
use crate::error::GrokOzempicError;
use crate::types::{
    GROK1_BLOCK_SLOTS, GROK1_HIDDEN_DIM, GROK1_TENSOR_F32, GROK1_TENSOR_INT8, GROK1_TENSOR_QUANT,
    GROK1_TENSOR_TOTAL, GROK1_TENSOR_TOTAL_BYTES, GROK1_TENSOR_TOTAL_ELEMENTS, GROK1_VOCAB_SIZE,
};
use std::fs;

pub(crate) fn grok1_spec_inventory_scan() -> InventoryScan {
    parse_inventory_scan_bytes(grok1_spec_inventory_json().as_bytes(), "<grok1-spec>")
        .expect("Grok-1 spec inventory must parse")
}

fn slot_json(slot: &crate::types::BlockSlot) -> String {
    let (role, dtype) = if slot.is_int8 {
        ("quant_weight", "i8")
    } else {
        ("tensor", "f32")
    };
    let shape = slot
        .shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"{{"role":"{role}","dtype":"{dtype}","shape":[{shape}],"nbytes":{}}}"#,
        slot.bytes
    )
}

fn grok1_spec_inventory_json() -> String {
    let mut tensors = Vec::with_capacity(GROK1_TENSOR_TOTAL);
    tensors.push(format!(
        r#"{{"role":"tensor","dtype":"f32","shape":[{}, {}],"nbytes":{}}}"#,
        GROK1_VOCAB_SIZE,
        GROK1_HIDDEN_DIM,
        GROK1_VOCAB_SIZE * GROK1_HIDDEN_DIM * 4
    ));
    for _ in 0..GROK1_BLOCK_COUNT {
        tensors.extend(GROK1_BLOCK_SLOTS.iter().map(slot_json));
    }
    tensors.push(format!(
        r#"{{"role":"tensor","dtype":"f32","shape":[{}],"nbytes":{}}}"#,
        GROK1_HIDDEN_DIM,
        GROK1_HIDDEN_DIM * 4
    ));
    format!(
        r#"{{
            "model_family": "grok-1",
            "checkpoint_path": "grok-1-official/ckpt-0",
            "shard_count": {GROK1_TENSOR_TOTAL},
            "tensors": [{}],
            "totals": {{
                "tensors": {GROK1_TENSOR_TOTAL},
                "quant_tensors": {GROK1_TENSOR_QUANT},
                "f32_tensors": {GROK1_TENSOR_F32},
                "i8_tensors": {GROK1_TENSOR_INT8},
                "total_nbytes": {GROK1_TENSOR_TOTAL_BYTES},
                "total_elements": {GROK1_TENSOR_TOTAL_ELEMENTS}
            }},
            "schema_version": 2
        }}"#,
        tensors.join(",")
    )
}

fn mini_inventory_json(totals_override: Option<&str>) -> String {
    let totals = totals_override.unwrap_or(
        r#"{
            "tensors": 2,
            "quant_tensors": 1,
            "f32_tensors": 1,
            "i8_tensors": 1,
            "total_nbytes": 21,
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
                    "role": "tensor",
                    "dtype": "f32",
                    "shape": [4],
                    "nbytes": 16
                }},
                {{
                    "shard_path": "/fixtures/t1",
                    "role": "quant_weight",
                    "dtype": "i8",
                    "shape": [5],
                    "nbytes": 5
                }}
            ],
            "blocks": [
                {{ "tensor_count": 1, "total_nbytes": 16 }},
                {{ "tensor_count": 1, "total_nbytes": 5 }}
            ],
            "totals": {totals},
            "schema_version": 2
        }}"#
    )
}

fn fixture_dir() -> std::path::PathBuf {
    let d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("scan-tests")
        .join(format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    fs::create_dir_all(&d).unwrap();
    d
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
    assert_eq!(scan.totals.total_bytes, 21);
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
            "total_nbytes": 21,
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
fn rejects_declared_subtotal_mismatches() {
    let json = mini_inventory_json(Some(
        r#"{
            "tensors": 2,
            "quant_tensors": 0,
            "f32_tensors": 0,
            "i8_tensors": 0,
            "total_nbytes": 0,
            "total_elements": 0
        }"#,
    ));
    let msg = parse_inventory_scan_bytes(json.as_bytes(), "<subs>")
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("f32_tensors: declared 0, derived 1"),
        "got {msg}"
    );
    assert!(
        msg.contains("i8_tensors: declared 0, derived 1"),
        "got {msg}"
    );
    assert!(
        msg.contains("quant_tensors: declared 0, derived 1"),
        "got {msg}"
    );
    assert!(
        msg.contains("total_elements: declared 0, derived 9"),
        "got {msg}"
    );
    assert!(
        msg.contains("total_nbytes: declared 0, derived 21"),
        "got {msg}"
    );
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
    let json = mini_inventory_json(None).replace("\"schema_version\": 2", "\"schema_version\": 1");
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
    assert_eq!(scan.shard_count, GROK1_TENSOR_TOTAL);
}

#[test]
fn load_inventory_scan_reads_a_file_and_rejects_missing_paths() {
    let dir = fixture_dir();
    let path = dir.join("inventory.json");
    fs::write(&path, mini_inventory_json(None)).expect("write fixture");
    let scan = load_inventory_scan(&path).expect("load fixture");
    assert_eq!(scan.totals.total, 2);
    fs::remove_file(&path).unwrap();
    let err = load_inventory_scan(&path).unwrap_err();
    assert!(
        matches!(err, GrokOzempicError::ManifestIo { .. }),
        "got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn rejects_malformed_json() {
    let err = parse_inventory_scan_bytes(b"{ not json", "<bad>").unwrap_err();
    assert!(
        matches!(err, GrokOzempicError::ManifestParse { .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_non_grok1_family() {
    let json = mini_inventory_json(None).replace("\"grok-1\"", "\"grok-2\"");
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<fam>").unwrap_err();
    assert!(
        err.to_string().contains("supports only grok-1"),
        "got {err}"
    );
}

#[test]
fn rejects_empty_tensors() {
    let json = r#"{
        "model_family": "grok-1",
        "checkpoint_path": "",
        "shard_count": 1,
        "tensors": [],
        "totals": {
            "tensors": 0, "quant_tensors": 0, "f32_tensors": 0,
            "i8_tensors": 0, "total_nbytes": 0, "total_elements": 0
        },
        "schema_version": 2
    }"#;
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<empty>").unwrap_err();
    assert!(
        err.to_string().contains("tensors` array is empty"),
        "got {err}"
    );
}

#[test]
fn rejects_zero_shard_count() {
    let json = mini_inventory_json(None).replace("\"shard_count\": 2", "\"shard_count\": 0");
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<zero>").unwrap_err();
    assert!(err.to_string().contains("shard_count is 0"), "got {err}");
}

#[test]
fn rejects_block_tensor_count_mismatch() {
    let json = mini_inventory_json(None).replace("\"tensor_count\": 1", "\"tensor_count\": 9");
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<blocks>").unwrap_err();
    assert!(
        err.to_string().contains("block summaries count"),
        "got {err}"
    );
}

#[test]
fn rejects_block_byte_sum_mismatch() {
    let json = mini_inventory_json(None).replace("\"total_nbytes\": 16", "\"total_nbytes\": 99");
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<bytes>").unwrap_err();
    assert!(err.to_string().contains("block summaries sum"), "got {err}");
}

#[test]
fn omitted_blocks_and_quant_scales_are_accepted() {
    let json = r#"{
        "model_family": "grok-1",
        "checkpoint_path": "/fixtures/ckpt-0",
        "shard_count": 2,
        "tensors": [
            {"role": "quant_scales", "dtype": "f32", "shape": [4], "nbytes": 16},
            {"role": "quant_weight", "dtype": "i8", "shape": [5], "nbytes": 5}
        ],
        "totals": {
            "tensors": 2, "quant_tensors": 2, "f32_tensors": 1,
            "i8_tensors": 1, "total_nbytes": 21, "total_elements": 9
        },
        "schema_version": 2
    }"#;
    let scan = parse_inventory_scan_bytes(json.as_bytes(), "<noblocks>")
        .expect("omitted blocks are advisory");
    assert_eq!(scan.totals.quant_tensors, 2);
    assert_eq!(scan.checkpoint_path, "/fixtures/ckpt-0");
    assert_eq!(scan.tensors[0].role, ScanRole::QuantScales);
}

#[test]
fn rejects_shape_element_count_overflow() {
    let json = r#"{
        "model_family": "grok-1",
        "checkpoint_path": "",
        "shard_count": 1,
        "tensors": [{
            "role": "tensor",
            "dtype": "i8",
            "shape": [18446744073709551615, 2],
            "nbytes": 0
        }],
        "totals": {
            "tensors": 1, "quant_tensors": 0, "f32_tensors": 0,
            "i8_tensors": 1, "total_nbytes": 0, "total_elements": 0
        },
        "schema_version": 2
    }"#;
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<ovf>").unwrap_err();
    assert!(
        err.to_string().contains("overflows u64 element count"),
        "got {err}"
    );
}

#[test]
fn rejects_nbytes_itemsize_overflow() {
    let json = r#"{
        "model_family": "grok-1",
        "checkpoint_path": "",
        "shard_count": 1,
        "tensors": [{
            "role": "tensor",
            "dtype": "f32",
            "shape": [4611686018427387904],
            "nbytes": 0
        }],
        "totals": {
            "tensors": 1, "quant_tensors": 0, "f32_tensors": 1,
            "i8_tensors": 0, "total_nbytes": 0, "total_elements": 0
        },
        "schema_version": 2
    }"#;
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<mul>").unwrap_err();
    assert!(
        err.to_string().contains("nbytes overflows u64"),
        "got {err}"
    );
}

#[test]
fn rejects_total_elements_overflow() {
    // Two i8 tensors of 2^63 elements: element sum overflows before bytes
    // would be distinguished, because nbytes == numel for i8.
    let json = r#"{
        "model_family": "grok-1",
        "checkpoint_path": "",
        "shard_count": 2,
        "tensors": [
            {"role": "tensor", "dtype": "i8", "shape": [9223372036854775808], "nbytes": 9223372036854775808},
            {"role": "tensor", "dtype": "i8", "shape": [9223372036854775808], "nbytes": 9223372036854775808}
        ],
        "totals": {
            "tensors": 2, "quant_tensors": 0, "f32_tensors": 0,
            "i8_tensors": 2, "total_nbytes": 0, "total_elements": 0
        },
        "schema_version": 2
    }"#;
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<elems>").unwrap_err();
    assert!(
        err.to_string().contains("total_elements overflows u64"),
        "got {err}"
    );
}

#[test]
fn rejects_total_bytes_overflow() {
    // Two f32 tensors of 2^61 elements: nbytes is 2^63 each, so bytes overflow
    // while the element sum (2^62) still fits in u64.
    let json = r#"{
        "model_family": "grok-1",
        "checkpoint_path": "",
        "shard_count": 2,
        "tensors": [
            {"role": "tensor", "dtype": "f32", "shape": [2305843009213693952], "nbytes": 9223372036854775808},
            {"role": "tensor", "dtype": "f32", "shape": [2305843009213693952], "nbytes": 9223372036854775808}
        ],
        "totals": {
            "tensors": 2, "quant_tensors": 0, "f32_tensors": 2,
            "i8_tensors": 0, "total_nbytes": 0, "total_elements": 0
        },
        "schema_version": 2
    }"#;
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<sum>").unwrap_err();
    assert!(
        err.to_string().contains("total_bytes overflows u64"),
        "got {err}"
    );
}

#[test]
fn rejects_block_nbytes_overflow() {
    let json = r#"{
        "model_family": "grok-1",
        "checkpoint_path": "",
        "shard_count": 2,
        "tensors": [
            {"role": "tensor", "dtype": "f32", "shape": [4], "nbytes": 16},
            {"role": "quant_weight", "dtype": "i8", "shape": [5], "nbytes": 5}
        ],
        "blocks": [
            {"tensor_count": 1, "total_nbytes": 9223372036854775808},
            {"tensor_count": 1, "total_nbytes": 9223372036854775808}
        ],
        "totals": {
            "tensors": 2, "quant_tensors": 1, "f32_tensors": 1,
            "i8_tensors": 1, "total_nbytes": 21, "total_elements": 9
        },
        "schema_version": 2
    }"#;
    let err = parse_inventory_scan_bytes(json.as_bytes(), "<blk>").unwrap_err();
    assert!(
        err.to_string().contains("block total_nbytes overflows u64"),
        "got {err}"
    );
}

#[test]
fn spec_scan_tensor_roles_follow_int8_flag() {
    let scan = grok1_spec_inventory_scan();
    let embedding = &scan.tensors[0];
    assert_eq!(embedding.dtype, ScanDtype::F32);
    assert_eq!(embedding.role, ScanRole::Tensor);
    let first_expert = &scan.tensors[1];
    assert_eq!(first_expert.dtype, ScanDtype::I8);
    assert_eq!(first_expert.role, ScanRole::QuantWeight);
}
