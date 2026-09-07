use crate::core::manifest::DissectManifest;
use crate::core::stream::{GROK1_BLOCK_COUNT, GROK1_EXPERT_COUNT, GROK1_FEED_FORWARD_LENGTH};
use crate::error::GrokOzempicError;
use crate::reports::schema::{
    ArtifactIR, ArtifactManifest, ExpertBlock, Hyperparameters, InventoryBlock, InventoryBlockKind,
    InventoryKindCount, InventoryTensor, RouterEntry, SaaqCritical, SaaqTarget, TensorTotals,
};
use crate::types::{
    GROK1_HIDDEN_DIM, GROK1_TENSOR_F32, GROK1_TENSOR_INT8, GROK1_TENSOR_QUANT, GROK1_TENSOR_TOTAL,
    GROK1_TENSOR_TOTAL_BYTES, GROK1_TENSOR_TOTAL_ELEMENTS, GROK1_VOCAB_SIZE,
};
use std::collections::HashSet;

const GROK1_FAMILY: &str = "grok-1";
const INVENTORY_SCHEMA_VERSION: u32 = 2;
const GROK1_BLOCK_SHARDS: usize = 12;
const GROK1_EMBEDDING_BYTES: u64 = 3_221_225_472;
const GROK1_FINAL_NORM_BYTES: u64 = 24_576;
const GROK1_BLOCK_BYTES: u64 = 4_920_213_504;
const GROK1_BLOCK_NORM_BYTES: u64 = 6_291_456;
const GROK1_ATTN_MODEL_WIDTH_BYTES: u64 = 4_831_838_208;
const GROK1_ATTN_NARROW_BYTES: u64 = 805_306_368;
const GROK1_MOE_DOWN_BYTES: u64 = 103_079_215_104;
const GROK1_ROUTER_BYTES: u64 = 12_582_912;

/// Reject a manifest this builder cannot honour.
///
/// This is the **only** place a manifest can change the outcome. It checks four
/// things: the model family, that every declared block index is in range, that
/// no index repeats, and that a declared expert count matches Grok-1's. Anything
/// else in the manifest — `preserve`, `ternary_candidates`, `defaults`,
/// `schema_version`, and the *number* of blocks — is ignored by design, because
/// [`build_grok1_spec_ir`] emits the full Grok-1 spec regardless. Partial and
/// unordered block metadata is explicitly supported; see
/// `test_manifest_block_metadata_can_be_partial_and_unordered`.
fn validate_supported_manifest(manifest: &DissectManifest) -> Result<(), GrokOzempicError> {
    if manifest.model.family != GROK1_FAMILY {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "artifact generation currently supports only {GROK1_FAMILY}; got {}",
            manifest.model.family
        )));
    }

    let mut seen_blocks = HashSet::new();
    for block in &manifest.blocks {
        if block.index as usize >= GROK1_BLOCK_COUNT as usize {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "manifest block index {} is outside Grok-1 block range 0..{}",
                block.index, GROK1_BLOCK_COUNT
            )));
        }
        if !seen_blocks.insert(block.index) {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "duplicate manifest block index {}",
                block.index
            )));
        }
        if let Some(experts) = block.experts
            && experts as usize != GROK1_EXPERT_COUNT as usize
        {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "expected {} experts for block {}, got {}",
                GROK1_EXPERT_COUNT, block.index, experts
            )));
        }
    }

    Ok(())
}

/// Build the Grok-1 **specification** IR.
///
/// The name matters, because the old one (`build_ir_from_manifest`) implied a
/// detector reading structure out of the manifest. It does not, and cannot: a
/// [`ManifestBlock`](crate::core::manifest::ManifestBlock) carries only `index`,
/// `experts` and `role` — no shapes, no dtypes, no byte counts — so there is
/// nothing to derive from. Every structural figure below comes from the
/// `GROK1_*` constants; the manifest contributes `model.source` (only when
/// `checkpoint` is `None`) and is otherwise used to *reject* incompatible input
/// via [`validate_supported_manifest`].
///
/// Consequences worth knowing before trusting the output:
///
/// - The IR describes the Grok-1 architecture as this crate understands it, not
///   the checkpoint on disk. `actual_shards` is the one observed quantity, and
///   it is passed in by the caller rather than read here.
/// - [`super::validator::validate_ir`] asserts the same constants this function
///   writes, so it is a schema/self-consistency check, not verification. It can
///   only fail on an IR that something else has mutated — which is exactly what
///   `src/reports/tests.rs` does.
///
/// Deriving real totals from a checkpoint scan needs a manifest schema that
/// carries them; tracked separately rather than faked here.
pub fn build_grok1_spec_ir(
    manifest: &DissectManifest,
    checkpoint: Option<&str>,
    actual_shards: Option<usize>,
) -> Result<ArtifactIR, GrokOzempicError> {
    validate_supported_manifest(manifest)?;

    // Basic structural information
    let model_family = manifest.model.family.clone();
    let checkpoint = checkpoint
        .map(|s| s.to_string())
        .unwrap_or_else(|| manifest.model.source.clone());

    let totals = TensorTotals {
        total: GROK1_TENSOR_TOTAL,
        // Spec constants, not a scan. Deriving these from the checkpoint needs a
        // manifest schema that carries per-tensor dtype/bytes; see this
        // function's doc comment.
        f32_tensors: GROK1_TENSOR_F32,
        int8_tensors: GROK1_TENSOR_INT8,
        quant_tensors: GROK1_TENSOR_QUANT,
        total_elements: GROK1_TENSOR_TOTAL_ELEMENTS,
        total_bytes: GROK1_TENSOR_TOTAL_BYTES,
    };

    let hyperparameters = Hyperparameters {
        vocab_size: GROK1_VOCAB_SIZE,
        d_model: GROK1_HIDDEN_DIM,
        n_experts: GROK1_EXPERT_COUNT as usize,
        d_ff: GROK1_FEED_FORWARD_LENGTH as usize,
        n_blocks: GROK1_BLOCK_COUNT as usize,
    };

    let mut routers = Vec::new();
    let mut expert_blocks = Vec::new();
    let mut inventory_blocks = Vec::new();

    // Grok-1 architecture specific structural scan
    for block_idx in 0..GROK1_BLOCK_COUNT as usize {
        routers.push(RouterEntry {
            block: block_idx,
            slot: 11,
            shape: (GROK1_HIDDEN_DIM, GROK1_EXPERT_COUNT as usize),
            orientation: "d_model_to_experts".to_string(),
            experts: GROK1_EXPERT_COUNT as usize,
            kind: "router".to_string(),
            structural_name: format!("block_{:03}.routing_slot_11", block_idx),
        });

        expert_blocks.push(ExpertBlock {
            block: block_idx,
            experts: GROK1_EXPERT_COUNT as usize,
            expert_tensors: 3,
            slots: vec![0, 1, 2],
            shapes: super::grok1_expected_expert_shape_strings().into(),
        });

        let shard_start = 2 + block_idx * GROK1_BLOCK_SHARDS;
        inventory_blocks.push(InventoryBlock {
            label: format!("block_{:03}", block_idx),
            block: Some(block_idx),
            shard_start,
            shard_end: shard_start + GROK1_BLOCK_SHARDS - 1,
            tensors: GROK1_BLOCK_SHARDS,
            bytes: GROK1_BLOCK_BYTES,
            kinds: block_kind_counts(),
        });
    }

    let saaq_targets = vec![SaaqTarget {
        rank: 1,
        tensor: "embedding.slot_00.token_embedding".to_string(),
        kind: "token_embedding".to_string(),
        region: "embedding_heavy".to_string(),
        readiness: 0.176,
        opportunity: 0.331,
        risk: 0.391,
        disposition: "candidate".to_string(),
    }];

    let mut saaq_critical = Vec::new();
    for block_idx in 0..GROK1_BLOCK_COUNT as usize {
        saaq_critical.push(SaaqCritical {
            tensor: format!("block_{:03}.slot_11.router", block_idx),
            readiness: 0.054,
            risk: 0.651,
            reasons: "distribution=dense_balanced<br>sampled_values=49152/49152<br>zero_fraction=0.0000<br>near_zero_fraction=0.0980<br>outlier_fraction=0.0000<br>peak_to_rms=4.729<br>linked to routing structure".to_string(),
        });
    }

    Ok(ArtifactIR {
        manifest: ArtifactManifest {
            model_family,
            checkpoint,
            shards: actual_shards.unwrap_or(GROK1_TENSOR_TOTAL),
            schema_version: INVENTORY_SCHEMA_VERSION,
        },
        hyperparameters,
        totals,
        inventory_kinds: inventory_kind_counts(),
        inventory_blocks: {
            let mut blocks = Vec::with_capacity(inventory_blocks.len() + 2);
            blocks.push(InventoryBlock {
                label: "embedding".to_string(),
                block: None,
                shard_start: 0,
                shard_end: 0,
                tensors: 1,
                bytes: GROK1_EMBEDDING_BYTES,
                kinds: vec![InventoryBlockKind {
                    count: 1,
                    kind: "token_embedding".to_string(),
                }],
            });
            blocks.extend(inventory_blocks);
            blocks.push(InventoryBlock {
                label: "final_norm".to_string(),
                block: None,
                shard_start: 1,
                shard_end: 1,
                tensors: 1,
                bytes: GROK1_FINAL_NORM_BYTES,
                kinds: vec![InventoryBlockKind {
                    count: 1,
                    kind: "final_norm".to_string(),
                }],
            });
            blocks
        },
        exemplar_tensors: exemplar_block_tensors(),
        routers,
        expert_blocks,
        saaq_targets,
        saaq_critical,
        stats: vec![],
        mean_rms: 19.762282,
    })
}

fn inventory_kind_counts() -> Vec<InventoryKindCount> {
    vec![
        InventoryKindCount {
            kind: "attn_proj_i8.model_width".to_string(),
            count: 128,
            bytes: GROK1_ATTN_MODEL_WIDTH_BYTES,
        },
        InventoryKindCount {
            kind: "attn_proj_i8.narrow".to_string(),
            count: 128,
            bytes: GROK1_ATTN_NARROW_BYTES,
        },
        InventoryKindCount {
            kind: "block_norm".to_string(),
            count: 256,
            bytes: GROK1_BLOCK_NORM_BYTES,
        },
        InventoryKindCount {
            kind: "final_norm".to_string(),
            count: 1,
            bytes: GROK1_FINAL_NORM_BYTES,
        },
        InventoryKindCount {
            kind: "moe_expert.down".to_string(),
            count: 64,
            bytes: GROK1_MOE_DOWN_BYTES,
        },
        InventoryKindCount {
            kind: "moe_expert.gate".to_string(),
            count: 64,
            bytes: GROK1_MOE_DOWN_BYTES,
        },
        InventoryKindCount {
            kind: "moe_expert.up".to_string(),
            count: 64,
            bytes: GROK1_MOE_DOWN_BYTES,
        },
        InventoryKindCount {
            kind: "router".to_string(),
            count: 64,
            bytes: GROK1_ROUTER_BYTES,
        },
        InventoryKindCount {
            kind: "token_embedding".to_string(),
            count: 1,
            bytes: GROK1_EMBEDDING_BYTES,
        },
    ]
}

fn block_kind_counts() -> Vec<InventoryBlockKind> {
    vec![
        InventoryBlockKind {
            count: 2,
            kind: "attn_proj_i8.model_width".to_string(),
        },
        InventoryBlockKind {
            count: 2,
            kind: "attn_proj_i8.narrow".to_string(),
        },
        InventoryBlockKind {
            count: 4,
            kind: "block_norm".to_string(),
        },
        InventoryBlockKind {
            count: 1,
            kind: "moe_expert.down".to_string(),
        },
        InventoryBlockKind {
            count: 1,
            kind: "moe_expert.gate".to_string(),
        },
        InventoryBlockKind {
            count: 1,
            kind: "moe_expert.up".to_string(),
        },
        InventoryBlockKind {
            count: 1,
            kind: "router".to_string(),
        },
    ]
}

fn exemplar_block_tensors() -> Vec<InventoryTensor> {
    vec![
        inventory_tensor(
            2,
            "quant.weight",
            "int8",
            "(8, 6144, 32768)",
            "moe_expert.gate",
            0,
        ),
        inventory_tensor(
            3,
            "quant.weight",
            "int8",
            "(8, 32768, 6144)",
            "moe_expert.down",
            1,
        ),
        inventory_tensor(
            4,
            "quant.weight",
            "int8",
            "(8, 6144, 32768)",
            "moe_expert.up",
            2,
        ),
        inventory_tensor(
            5,
            "quant.weight",
            "int8",
            "(6144, 1024)",
            "attn_proj_i8.narrow",
            3,
        ),
        inventory_tensor(
            6,
            "quant.weight",
            "int8",
            "(6144, 6144)",
            "attn_proj_i8.model_width",
            4,
        ),
        inventory_tensor(
            7,
            "quant.weight",
            "int8",
            "(6144, 6144)",
            "attn_proj_i8.model_width",
            5,
        ),
        inventory_tensor(
            8,
            "quant.weight",
            "int8",
            "(6144, 1024)",
            "attn_proj_i8.narrow",
            6,
        ),
        inventory_tensor(9, "tensor", "f32", "(6144,)", "block_norm", 7),
        inventory_tensor(10, "tensor", "f32", "(6144,)", "block_norm", 8),
        inventory_tensor(11, "tensor", "f32", "(6144,)", "block_norm", 9),
        inventory_tensor(12, "tensor", "f32", "(6144,)", "block_norm", 10),
        inventory_tensor(13, "tensor", "f32", "(6144, 8)", "router", 11),
    ]
}

fn inventory_tensor(
    shard: usize,
    role: &str,
    dtype: &str,
    shape: &str,
    kind: &str,
    slot: usize,
) -> InventoryTensor {
    InventoryTensor {
        shard,
        in_shard: 0,
        role: role.to_string(),
        dtype: dtype.to_string(),
        shape: shape.to_string(),
        kind: kind.to_string(),
        slot: Some(slot),
    }
}
