use crate::core::manifest::ManifestBlock;
use crate::core::manifest::embedded_grok1_baseline;
use crate::reports::detector;
use crate::reports::schema::ArtifactIR;
use crate::reports::templates;
use crate::reports::validator;

fn create_valid_ir() -> ArtifactIR {
    // We can use the detector to build a valid IR from the embedded baseline
    let manifest = embedded_grok1_baseline().expect("Failed to get baseline manifest");
    detector::build_grok1_spec_ir(manifest, None, None).expect("Failed to build valid IR")
}

#[test]
fn test_inventory_totals() {
    let mut ir = create_valid_ir();
    assert_eq!(ir.totals.total, 770);
    assert_eq!(ir.totals.total_elements, 315_684_820_992);
    assert_eq!(ir.totals.total_bytes, 318_114_914_304);

    // Break the total while keeping subtotals internally consistent.
    ir.totals.total = 769;
    ir.totals.int8_tensors = 447;
    let res = validator::validate_ir(&ir);
    assert!(res.is_err());
    assert!(
        res.unwrap_err()
            .to_string()
            .contains("Total tensors mismatch")
    );
}

#[test]
fn test_inventory_uses_resolved_attention_labels() {
    let ir = create_valid_ir();

    assert_eq!(ir.manifest.schema_version, 2);
    assert!(validator::validate_ir(&ir).is_ok());

    let inventory = templates::generate_inventory(&ir);
    assert!(inventory.contains("| attn_proj_i8.model_width | 128 |"));
    assert!(inventory.contains("| attn_proj_i8.narrow | 128 |"));
    assert!(inventory.contains("2xattn_proj_i8.model_width"));
    assert!(inventory.contains("2xattn_proj_i8.narrow"));
    assert!(!inventory.contains("| unknown |"));
}

#[test]
fn test_scan_shard_count_does_not_override_tensor_totals() {
    let manifest = embedded_grok1_baseline().expect("Failed to get baseline manifest");
    let ir = detector::build_grok1_spec_ir(manifest, None, Some(42))
        .expect("Failed to build IR with scan-derived shard count");

    assert_eq!(ir.manifest.shards, 42);
    assert_eq!(ir.totals.total, 770);
    assert!(validator::validate_ir(&ir).is_ok());
}

#[test]
fn test_manifest_block_metadata_can_be_partial_and_unordered() {
    let mut manifest = embedded_grok1_baseline()
        .expect("Failed to get baseline manifest")
        .clone();
    manifest.blocks = vec![
        ManifestBlock {
            index: 7,
            experts: Some(8),
            role: Some("moe".to_string()),
        },
        ManifestBlock {
            index: 0,
            experts: Some(8),
            role: Some("moe".to_string()),
        },
    ];

    let ir = detector::build_grok1_spec_ir(&manifest, None, None)
        .expect("partial unordered advisory blocks should be accepted");
    assert!(validator::validate_ir(&ir).is_ok());

    // The consequence worth pinning: a manifest declaring TWO blocks still
    // yields the full 64-block spec IR. Block metadata is advisory -- the
    // builder emits the Grok-1 spec regardless -- so the block *count* is
    // ignored rather than rejected. Anyone reading the IR as a description of
    // the manifest, rather than of the architecture, would be wrong.
    assert_eq!(manifest.blocks.len(), 2, "fixture declares two blocks");
    assert_eq!(
        ir.hyperparameters.n_blocks, 64,
        "block count comes from the spec constant, not the manifest"
    );
    assert_eq!(
        ir.routers.len(),
        64,
        "one router per spec block, not per manifest block"
    );
}

#[test]
fn test_router_shape_strict() {
    let mut ir = create_valid_ir();

    // Inject invalid shape
    ir.routers[0].shape = (8, 6144); // Reversed
    let res = validator::validate_ir(&ir);
    assert!(res.is_err());
    assert!(
        res.unwrap_err()
            .to_string()
            .contains("Invalid router shape")
    );
}

#[test]
fn test_expert_slot_mapping() {
    let mut ir = create_valid_ir();

    // Mess up the order
    ir.expert_blocks[0].shapes[0] = "expert_slot_00 (8, 32768, 6144)".to_string(); // Wrong shape for slot 00

    let res = validator::validate_ir(&ir);
    assert!(res.is_err());
    assert!(
        res.unwrap_err()
            .to_string()
            .contains("Invalid expert shape")
    );
}

#[test]
fn test_saaq_readiness_criticality() {
    let mut ir = create_valid_ir();

    // The valid IR should pass
    assert!(validator::validate_ir(&ir).is_ok());

    // Remove token_embedding candidate
    ir.saaq_targets.clear();
    let res = validator::validate_ir(&ir);
    assert!(res.is_err());
    assert!(
        res.unwrap_err()
            .to_string()
            .contains("Missing token_embedding")
    );

    // Reset and break router criticality
    let mut ir = create_valid_ir();
    ir.saaq_critical.pop();
    let res = validator::validate_ir(&ir);
    assert!(res.is_err());
    assert!(
        res.unwrap_err()
            .to_string()
            .contains("Missing high-risk critical routers")
    );
}

/// The detector's per-block kind counts must be derivable from the shared
/// `GROK1_BLOCK_SLOTS` table (GH #106).
///
/// This is the third of the three tables that had drifted, and the one the
/// cross-check in `grok1_inventory.rs` does not reach: `block_kind_counts()` is
/// private to `detector`, so this asserts on its observable output in the built
/// IR instead. Without it, `detector` could go back to saying
/// `moe_expert.unresolved` while the shared table said `.gate`/`.up`, and only
/// the two other copies would be pinned.
#[test]
fn detector_block_kind_counts_match_the_shared_slot_table() {
    use crate::types::GROK1_BLOCK_SLOTS;
    use std::collections::BTreeMap;

    let ir = create_valid_ir();

    // Expected counts derived from the single source, not retyped.
    let mut expected: BTreeMap<&str, usize> = BTreeMap::new();
    for slot in GROK1_BLOCK_SLOTS.iter() {
        *expected.entry(slot.kind).or_default() += 1;
    }

    let block = ir
        .inventory_blocks
        .iter()
        .find(|b| b.block == Some(0))
        .expect("block 000 inventory entry");

    let actual: BTreeMap<&str, usize> = block
        .kinds
        .iter()
        .map(|k| (k.kind.as_str(), k.count))
        .collect();

    assert_eq!(
        actual, expected,
        "detector block kind counts must match GROK1_BLOCK_SLOTS; a divergence \
         here is the drift GH #106 fixed reappearing in the third copy"
    );

    // The counts must also sum to the slot count, so a dropped kind cannot hide
    // behind another kind being inflated.
    let total: usize = actual.values().sum();
    assert_eq!(
        total,
        GROK1_BLOCK_SLOTS.len(),
        "per-block kinds must account for every slot"
    );

    // And the resolved MoE labels specifically, since those are what drifted.
    assert_eq!(actual.get("moe_expert.gate"), Some(&1));
    assert_eq!(actual.get("moe_expert.up"), Some(&1));
    assert!(
        !actual.keys().any(|k| k.contains("unresolved")),
        "detector must not reintroduce moe_expert.unresolved"
    );

    // The GLOBAL kind counts are a second hardcoded table in the same file, and
    // they are exactly the per-block counts times the block count, plus the two
    // non-block tensors. Deriving them here means the per-block and whole-model
    // tables cannot disagree.
    //
    // Pinning this was prompted by a mutation that did not fail: reverting a
    // label inside `inventory_kind_counts()` left the per-block assertion above
    // green, because that assertion only reaches `block_kind_counts()`.
    let n_blocks = ir.hyperparameters.n_blocks;
    let mut expected_global: BTreeMap<&str, usize> =
        expected.iter().map(|(k, v)| (*k, v * n_blocks)).collect();
    expected_global.insert("token_embedding", 1);
    expected_global.insert("final_norm", 1);

    let actual_global: BTreeMap<&str, usize> = ir
        .inventory_kinds
        .iter()
        .map(|k| (k.kind.as_str(), k.count))
        .collect();

    assert_eq!(
        actual_global, expected_global,
        "whole-model kind counts must be the per-slot table scaled by n_blocks \
         plus the embedding and final norm"
    );
}
