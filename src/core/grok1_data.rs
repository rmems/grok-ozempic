//! Auto-generated Grok-1 tensor inventory data.
//! This file is included by grok1_inventory.rs to keep the main file small.

use crate::core::inventory::InventoryTensor;
use crate::core::selection::TensorClass;
use crate::core::stream::GROK1_BLOCK_COUNT;

/// Number of blocks in the Grok-1 architecture (matches HF config.json).
pub const GROK1_BLOCKS: u32 = GROK1_BLOCK_COUNT;

fn create_embedding_tensor() -> InventoryTensor {
    InventoryTensor {
        structural_name: "embedding.slot_00.token_embedding".into(),
        expected_class: TensorClass::TernaryCandidate {
            rank: None,
            gif_threshold: None,
        },
        dtype: "f32",
        block: None,
        slot: None,
        kind: "token_embedding",
    }
}

fn create_final_norm_tensor() -> InventoryTensor {
    InventoryTensor {
        structural_name: "final_norm.slot_00.final_norm".into(),
        expected_class: TensorClass::Preserve {
            reason: Some("normalization-critical; must remain FP32".into()),
        },
        dtype: "f32",
        block: None,
        slot: None,
        kind: "final_norm",
    }
}

fn create_moe_expert_tensors(blk: u32) -> Vec<InventoryTensor> {
    let b = blk;
    vec![
        InventoryTensor {
            structural_name: format!("block_{blk:03}.slot_00.moe_expert.gate"),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype: "i8",
            block: Some(b),
            slot: Some(0),
            kind: "moe_expert.gate",
        },
        InventoryTensor {
            structural_name: format!("block_{blk:03}.slot_01.moe_expert.down"),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype: "i8",
            block: Some(b),
            slot: Some(1),
            kind: "moe_expert.down",
        },
        InventoryTensor {
            structural_name: format!("block_{blk:03}.slot_02.moe_expert.up"),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype: "i8",
            block: Some(b),
            slot: Some(2),
            kind: "moe_expert.up",
        },
    ]
}

fn create_attn_proj_i8_tensors(blk: u32) -> Vec<InventoryTensor> {
    let b = blk;
    vec![
        InventoryTensor {
            structural_name: format!("block_{blk:03}.slot_03.attn_proj_i8.narrow"),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype: "i8",
            block: Some(b),
            slot: Some(3),
            kind: "attn_proj_i8.narrow",
        },
        InventoryTensor {
            structural_name: format!("block_{blk:03}.slot_04.attn_proj_i8.model_width"),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype: "i8",
            block: Some(b),
            slot: Some(4),
            kind: "attn_proj_i8.model_width",
        },
        InventoryTensor {
            structural_name: format!("block_{blk:03}.slot_05.attn_proj_i8.model_width"),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype: "i8",
            block: Some(b),
            slot: Some(5),
            kind: "attn_proj_i8.model_width",
        },
        InventoryTensor {
            structural_name: format!("block_{blk:03}.slot_06.attn_proj_i8.narrow"),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype: "i8",
            block: Some(b),
            slot: Some(6),
            kind: "attn_proj_i8.narrow",
        },
    ]
}

fn create_block_norm_tensors(blk: u32) -> Vec<InventoryTensor> {
    let b = blk;
    (7..11)
        .map(|slot| InventoryTensor {
            structural_name: format!("block_{blk:03}.slot_{slot:02}.block_norm"),
            expected_class: TensorClass::Preserve {
                reason: Some("normalization-critical; must remain FP32".into()),
            },
            dtype: "f32",
            block: Some(b),
            slot: Some(slot as u32),
            kind: "block_norm",
        })
        .collect()
}

fn create_router_tensor(blk: u32) -> InventoryTensor {
    InventoryTensor {
        structural_name: format!("block_{blk:03}.slot_11.router"),
        expected_class: TensorClass::Preserve {
            reason: Some("routing-critical; quantization changes expert selection".into()),
        },
        dtype: "f32",
        block: Some(blk),
        slot: Some(11),
        kind: "router",
    }
}

pub fn build_grok1_tensors() -> Vec<InventoryTensor> {
    let mut tensors = Vec::with_capacity(770);

    tensors.push(create_embedding_tensor());
    tensors.push(create_final_norm_tensor());

    for blk in 0..GROK1_BLOCKS {
        tensors.extend(create_moe_expert_tensors(blk));
        tensors.extend(create_attn_proj_i8_tensors(blk));
        tensors.extend(create_block_norm_tensors(blk));
        tensors.push(create_router_tensor(blk));
    }

    tensors
}
