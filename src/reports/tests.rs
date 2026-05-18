use crate::core::manifest::ManifestBlock;
use crate::core::manifest::embedded_grok1_baseline;
use crate::reports::detector;
use crate::reports::schema::ArtifactIR;
use crate::reports::templates;
use crate::reports::validator;

fn create_valid_ir() -> ArtifactIR {
    // We can use the detector to build a valid IR from the embedded baseline
    let manifest = embedded_grok1_baseline().expect("Failed to get baseline manifest");
    detector::build_ir_from_manifest(manifest, None, None).expect("Failed to build valid IR")
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
    let ir = detector::build_ir_from_manifest(manifest, None, Some(42))
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

    let ir = detector::build_ir_from_manifest(&manifest, None, None)
        .expect("partial unordered advisory blocks should be accepted");
    assert!(validator::validate_ir(&ir).is_ok());
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
