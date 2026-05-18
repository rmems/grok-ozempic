use crate::error::GrokOzempicError;
use crate::reports::schema::ArtifactIR;
use std::collections::HashSet;

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

    let mut seen_router_blocks = HashSet::new();
    for router in &ir.routers {
        if router.block >= ir.hyperparameters.n_blocks {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router block index: expected < {}, got {}",
                ir.hyperparameters.n_blocks, router.block
            )));
        }
        if !seen_router_blocks.insert(router.block) {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Duplicate router entry for block {}",
                router.block
            )));
        }

        let expected_router_shape = (ir.hyperparameters.d_model, ir.hyperparameters.n_experts);
        if router.shape != expected_router_shape {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router shape for block {}: expected {:?}, got {:?}",
                router.block, expected_router_shape, router.shape
            )));
        }

        if router.orientation != "d_model_to_experts" {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router orientation for block {}: expected d_model_to_experts, got {}",
                router.block, router.orientation
            )));
        }

        if router.slot != 11 {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router slot for block {}: expected 11, got {}",
                router.block, router.slot
            )));
        }

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

    let mut seen_expert_blocks = HashSet::new();
    for block in &ir.expert_blocks {
        if block.block >= ir.hyperparameters.n_blocks {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid expert block index: expected < {}, got {}",
                ir.hyperparameters.n_blocks, block.block
            )));
        }
        if !seen_expert_blocks.insert(block.block) {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Duplicate expert block entry for block {}",
                block.block
            )));
        }
        if block.expert_tensors != 3 {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid expert tensors count for block {}: expected 3, got {}",
                block.block, block.expert_tensors
            )));
        }

        if block.experts != ir.hyperparameters.n_experts {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid expert count for block {}: expected {}, got {}",
                block.block, ir.hyperparameters.n_experts, block.experts
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
                "Invalid expert shape count for block {}: expected {}, got {}",
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

    let mut critical_router_blocks = HashSet::new();
    for c in &ir.saaq_critical {
        if let Some(block_str) = c
            .tensor
            .strip_prefix("block_")
            .and_then(|s| s.split_once('.'))
            .map(|(prefix, _)| prefix)
        {
            if let Ok(block) = block_str.parse::<usize>() {
                if block < ir.hyperparameters.n_blocks
                    && c.tensor.ends_with("slot_11.router")
                    && c.risk > 0.5
                {
                    critical_router_blocks.insert(block);
                }
            }
        }
    }

    if critical_router_blocks.len() != ir.hyperparameters.n_blocks {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "Missing high-risk critical routers: expected {}, got {}",
            ir.hyperparameters.n_blocks,
            critical_router_blocks.len()
        )));
    }

    Ok(())
}
