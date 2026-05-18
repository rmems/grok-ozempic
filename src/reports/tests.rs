use crate::core::manifest::embedded_grok1_baseline;
use crate::reports::detector;
use crate::reports::schema::ArtifactIR;
use crate::reports::validator;

fn create_valid_ir() -> ArtifactIR {
    // We can use the detector to build a valid IR from the embedded baseline
    let manifest = embedded_grok1_baseline().expect("Failed to get baseline manifest");
    detector::build_ir_from_manifest(manifest).expect("Failed to build valid IR")
}

#[test]
fn test_inventory_totals() {
    let mut ir = create_valid_ir();
    assert_eq!(ir.totals.total, 770);

    // Break the total
    ir.totals.total = 769;
    let res = validator::validate_ir(&ir);
    assert!(res.is_err());
    assert!(
        res.unwrap_err()
            .to_string()
            .contains("Total tensors mismatch")
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
