use crate::core::stream::{GROK1_EXPERT_COUNT, GROK1_FEED_FORWARD_LENGTH};
use crate::error::GrokOzempicError;
use crate::reports::schema::ArtifactIR;
use crate::types::{
    GROK1_HIDDEN_DIM, GROK1_TENSOR_F32, GROK1_TENSOR_INT8, GROK1_TENSOR_QUANT, GROK1_TENSOR_TOTAL,
    GROK1_TENSOR_TOTAL_BYTES, GROK1_TENSOR_TOTAL_ELEMENTS,
};
use std::collections::HashSet;
const CRITICAL_ROUTER_RISK_THRESHOLD: f64 = 0.5;

pub fn validate_ir(ir: &ArtifactIR) -> Result<(), GrokOzempicError> {
    if ir.hyperparameters.d_model != GROK1_HIDDEN_DIM {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "d_model mismatch: expected {}, got {}",
            GROK1_HIDDEN_DIM, ir.hyperparameters.d_model
        )));
    }
    if ir.hyperparameters.n_experts != GROK1_EXPERT_COUNT as usize {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "n_experts mismatch: expected {}, got {}",
            GROK1_EXPERT_COUNT, ir.hyperparameters.n_experts
        )));
    }

    // 1. Tensor inventory totals (strict Grok-1 baseline + internal consistency).
    let sum_f32_int8 = ir
        .totals
        .f32_tensors
        .checked_add(ir.totals.int8_tensors)
        .ok_or_else(|| {
            GrokOzempicError::ArtifactValidation(
                "Tensor subtotals overflow when computing f32_tensors + int8_tensors".to_string(),
            )
        })?;
    if sum_f32_int8 != ir.totals.total {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "Tensor subtotals do not sum to total: f32_tensors ({}) + int8_tensors ({}) = {}, expected total {}",
            ir.totals.f32_tensors, ir.totals.int8_tensors, sum_f32_int8, ir.totals.total
        )));
    }
    if ir.totals.total != GROK1_TENSOR_TOTAL {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "Total tensors mismatch: expected {}, got {}",
            GROK1_TENSOR_TOTAL, ir.totals.total
        )));
    }
    if ir.totals.f32_tensors != GROK1_TENSOR_F32 {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "f32 tensor count mismatch: expected {}, got {}",
            GROK1_TENSOR_F32, ir.totals.f32_tensors
        )));
    }
    if ir.totals.int8_tensors != GROK1_TENSOR_INT8 {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "int8 tensor count mismatch: expected {}, got {}",
            GROK1_TENSOR_INT8, ir.totals.int8_tensors
        )));
    }
    if ir.totals.quant_tensors != GROK1_TENSOR_QUANT {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "quant tensor count mismatch: expected {}, got {}",
            GROK1_TENSOR_QUANT, ir.totals.quant_tensors
        )));
    }
    if ir.totals.total_elements != GROK1_TENSOR_TOTAL_ELEMENTS {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "total element count mismatch: expected {}, got {}",
            GROK1_TENSOR_TOTAL_ELEMENTS, ir.totals.total_elements
        )));
    }
    if ir.totals.total_bytes != GROK1_TENSOR_TOTAL_BYTES {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "total byte count mismatch: expected {}, got {}",
            GROK1_TENSOR_TOTAL_BYTES, ir.totals.total_bytes
        )));
    }
    if ir.inventory_kinds.iter().any(|kind| kind.kind == "unknown") {
        return Err(GrokOzempicError::ArtifactValidation(
            "Inventory kind counts must not include unknown after xai-dissect schema v2 attention classification".to_string(),
        ));
    }
    for required in ["attn_proj_i8.model_width", "attn_proj_i8.narrow"] {
        if !ir.inventory_kinds.iter().any(|kind| kind.kind == required) {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Missing inventory kind count for {required}"
            )));
        }
    }

    // 2. Routing Detection Rules & Invariants
    if ir.hyperparameters.n_blocks != crate::core::stream::GROK1_BLOCK_COUNT as usize {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "n_blocks mismatch: expected {}, got {}",
            crate::core::stream::GROK1_BLOCK_COUNT,
            ir.hyperparameters.n_blocks
        )));
    }
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

        if router.experts != ir.hyperparameters.n_experts {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid router experts count for block {}: expected {}, got {}",
                router.block, ir.hyperparameters.n_experts, router.experts
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

    let expected_expert_shapes: [String; 3] = [
        format!(
            "expert_slot_00 ({}, {}, {})",
            GROK1_EXPERT_COUNT, GROK1_HIDDEN_DIM, GROK1_FEED_FORWARD_LENGTH
        ),
        format!(
            "expert_slot_01 ({}, {}, {})",
            GROK1_EXPERT_COUNT, GROK1_FEED_FORWARD_LENGTH, GROK1_HIDDEN_DIM
        ),
        format!(
            "expert_slot_02 ({}, {}, {})",
            GROK1_EXPERT_COUNT, GROK1_HIDDEN_DIM, GROK1_FEED_FORWARD_LENGTH
        ),
    ];

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

        if block.shapes.len() != expected_expert_shapes.len() {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "Invalid expert shape count for block {}: expected {}, got {}",
                block.block,
                expected_expert_shapes.len(),
                block.shapes.len()
            )));
        }

        for (i, shape) in block.shapes.iter().enumerate() {
            if shape != &expected_expert_shapes[i] {
                return Err(GrokOzempicError::ArtifactValidation(format!(
                    "Invalid expert shape for block {}, slot {}: expected {}, got {}",
                    block.block, i, expected_expert_shapes[i], shape
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
            && let Ok(block) = block_str.parse::<usize>()
            && block < ir.hyperparameters.n_blocks
            && c.tensor.ends_with("slot_11.router")
            && c.risk > CRITICAL_ROUTER_RISK_THRESHOLD
        {
            critical_router_blocks.insert(block);
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
