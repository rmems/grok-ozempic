use crate::error::GrokOzempicError;
use crate::reports::schema::ArtifactIR;

pub fn validate_ir(ir: &ArtifactIR) -> Result<(), GrokOzempicError> {
    // 1. Verify tensors parse and total exactly 770.
    if ir.totals.total != 770 {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "Total tensors mismatch: expected 770, got {}",
            ir.totals.total
        )));
    }

    // 2. Routing Detection Rules & Invariants
    if ir.routers.len() != ir.hyperparameters.n_blocks {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "Router count mismatch: expected {}, got {}",
            ir.hyperparameters.n_blocks,
            ir.routers.len()
        )));
    }

    for router in &ir.routers {
        // The shape must strictly be `(6144, 8)`.
        if router.shape != (6144, 8) {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router shape for block {}: expected (6144, 8), got {:?}",
                router.block, router.shape
            )));
        }

        // The orientation must be labeled `d_model_to_experts`.
        if router.orientation != "d_model_to_experts" {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router orientation for block {}: expected d_model_to_experts, got {}",
                router.block, router.orientation
            )));
        }

        // The artifact slot must be `11`.
        if router.slot != 11 {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router slot for block {}: expected 11, got {}",
                router.block, router.slot
            )));
        }

        // The structural name format must strictly be `block_{:03}.routing_slot_11`.
        let expected_name = format!("block_{:03}.routing_slot_11", router.block);
        if router.structural_name != expected_name {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router structural name for block {}: expected {}, got {}",
                router.block, expected_name, router.structural_name
            )));
        }
    }

    // 3. Expert Detection Rules & Invariants
    if ir.expert_blocks.len() != ir.hyperparameters.n_blocks {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "Expert block count mismatch: expected {}, got {}",
            ir.hyperparameters.n_blocks,
            ir.expert_blocks.len()
        )));
    }

    for block in &ir.expert_blocks {
        // Each block must have exactly 3 expert tensors.
        if block.expert_tensors != 3 {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid expert tensors count for block {}: expected 3, got {}",
                block.block, block.expert_tensors
            )));
        }

        // Total expert count per block must be `8`.
        if block.experts != 8 {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid expert count for block {}: expected 8, got {}",
                block.block, block.experts
            )));
        }

        if block.slots != vec![0, 1, 2] {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid expert slots for block {}: expected [0, 1, 2], got {:?}",
                block.block, block.slots
            )));
        }

        let expected_shapes = [
            "expert_slot_00 (8, 6144, 32768)",
            "expert_slot_01 (8, 32768, 6144)",
            "expert_slot_02 (8, 6144, 32768)",
        ];

        if block.shapes.len() != expected_shapes.len() {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid expert shapes count for block {}: expected {}, got {}",
                block.block,
                expected_shapes.len(),
                block.shapes.len()
            )));
        }

        for (i, shape) in block.shapes.iter().enumerate() {
            if shape != expected_shapes[i] {
                return Err(GrokOzempicError::ArtifactValidation(format!(
                    "Invalid expert shape for block {}, slot {}: expected {}, got {}",
                    block.block, i, expected_shapes[i], shape
                )));
            }
        }
    }

    // 4. SAAQ Readiness Criticality
    let has_token_embedding_target = ir
        .saaq_targets
        .iter()
        .any(|t| t.tensor == "embedding.slot_00.token_embedding" && t.disposition == "candidate");
    if !has_token_embedding_target {
        return Err(GrokOzempicError::ArtifactValidation(
            "Missing token_embedding as candidate target in saaq_targets".to_string(),
        ));
    }

    let mut router_critical_count = 0;
    for c in &ir.saaq_critical {
        if c.tensor.contains("slot_11.router") && c.risk > 0.5 {
            router_critical_count += 1;
        }
    }

    if router_critical_count != ir.hyperparameters.n_blocks {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "Missing high-risk critical routers: expected {}, got {}",
            ir.hyperparameters.n_blocks, router_critical_count
        )));
    }

    Ok(())
}
