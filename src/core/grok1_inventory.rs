use crate::core::selection::TensorClass;
use crate::core::stream::GROK1_BLOCK_COUNT;

pub const GROK1_BLOCKS: u32 = GROK1_BLOCK_COUNT;
pub const GROK1_NORMS_PER_BLOCK: usize = 4;
pub const GROK1_EXPERT_PROJECTIONS_PER_BLOCK: usize = 3;
pub const GROK1_ATTN_PROJECTIONS_PER_BLOCK: usize = 4;
pub const GROK1_ROUTERS_PER_BLOCK: usize = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct InventoryTensor {
    pub structural_name: String,
    pub expected_class: TensorClass,
    pub dtype: &'static str,
    pub block: Option<u32>,
    pub slot: Option<u32>,
    pub kind: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Grok1Inventory {
    pub tensors: Vec<InventoryTensor>,
}

impl Grok1Inventory {
    pub fn full() -> Self {
        let mut tensors = Vec::with_capacity(770);

        tensors.push(InventoryTensor {
            structural_name: "embedding.slot_00.token_embedding".into(),
            expected_class: TensorClass::TernaryCandidate {
                rank: None,
                gif_threshold: None,
            },
            dtype: "f32",
            block: None,
            slot: None,
            kind: "token_embedding",
        });

        tensors.push(InventoryTensor {
            structural_name: "final_norm.slot_00.final_norm".into(),
            expected_class: TensorClass::Preserve {
                reason: Some("normalization-critical; must remain FP32".into()),
            },
            dtype: "f32",
            block: None,
            slot: None,
            kind: "final_norm",
        });

        // NOTE (addressing Codex review on PR #26): The 448 i8 tensors (192 moe_expert + 256 attn_proj_i8)
        // are marked TernaryCandidate here + via the structural manifest globs to match the xai-dissect
        // structural inventory *exactly*. Alignment + classify_full_inventory tests require 0 defaults
        // and exact 448 ternary for full 770 coverage. The *streaming* path (build_manifest_* in stream.rs)
        // skips SourceDtype::Other (see parse_safetensors_dtype + npy_dtype_to_source; only F32/F16/BF16).
        // i8 data enters via xai-dissect "exports" + wrapping in src/artifact.rs (wrap_existing_int8_expert,
        // wrap_existing_int8_unknown etc.) + report validation (reports/validator.rs etc.).
        // This manifest declares *logical classification per xai-dissect*, not "all these will be
        // float-streamed from raw ckpt". Removing the i8 patterns would regress the purpose of this PR
        // (full inventory alignment verification). See also structural-manifest.json _i8_streaming_note.
        // Kilo agent xAI/Grok Build 0.1
        for blk in 0..GROK1_BLOCKS {
            let b = blk;

            tensors.push(InventoryTensor {
                structural_name: format!("block_{blk:03}.slot_00.moe_expert.gate"),
                expected_class: TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
                dtype: "i8",
                block: Some(b),
                slot: Some(0),
                kind: "moe_expert.gate",
            });

            tensors.push(InventoryTensor {
                structural_name: format!("block_{blk:03}.slot_01.moe_expert.down"),
                expected_class: TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
                dtype: "i8",
                block: Some(b),
                slot: Some(1),
                kind: "moe_expert.down",
            });

            tensors.push(InventoryTensor {
                structural_name: format!("block_{blk:03}.slot_02.moe_expert.up"),
                expected_class: TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
                dtype: "i8",
                block: Some(b),
                slot: Some(2),
                kind: "moe_expert.up",
            });

            tensors.push(InventoryTensor {
                structural_name: format!("block_{blk:03}.slot_03.attn_proj_i8.narrow"),
                expected_class: TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
                dtype: "i8",
                block: Some(b),
                slot: Some(3),
                kind: "attn_proj_i8.narrow",
            });

            tensors.push(InventoryTensor {
                structural_name: format!("block_{blk:03}.slot_04.attn_proj_i8.model_width"),
                expected_class: TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
                dtype: "i8",
                block: Some(b),
                slot: Some(4),
                kind: "attn_proj_i8.model_width",
            });

            tensors.push(InventoryTensor {
                structural_name: format!("block_{blk:03}.slot_05.attn_proj_i8.model_width"),
                expected_class: TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
                dtype: "i8",
                block: Some(b),
                slot: Some(5),
                kind: "attn_proj_i8.model_width",
            });

            tensors.push(InventoryTensor {
                structural_name: format!("block_{blk:03}.slot_06.attn_proj_i8.narrow"),
                expected_class: TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
                dtype: "i8",
                block: Some(b),
                slot: Some(6),
                kind: "attn_proj_i8.narrow",
            });

            for slot in 7..11 {
                tensors.push(InventoryTensor {
                    structural_name: format!("block_{blk:03}.slot_{slot:02}.block_norm"),
                    expected_class: TensorClass::Preserve {
                        reason: Some("normalization-critical; must remain FP32".into()),
                    },
                    dtype: "f32",
                    block: Some(b),
                    slot: Some(slot as u32),
                    kind: "block_norm",
                });
            }

            tensors.push(InventoryTensor {
                structural_name: format!("block_{blk:03}.slot_11.router"),
                expected_class: TensorClass::Preserve {
                    reason: Some("routing-critical; quantization changes expert selection".into()),
                },
                dtype: "f32",
                block: Some(b),
                slot: Some(11),
                kind: "router",
            });
        }

        Self { tensors }
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn count_by_expected_class(&self) -> (usize, usize, usize, usize) {
        let mut preserve = 0;
        let mut fp16 = 0;
        let mut ternary = 0;
        let mut default = 0;
        for t in &self.tensors {
            match &t.expected_class {
                TensorClass::Preserve { .. } => preserve += 1,
                TensorClass::Fp16 { .. } => fp16 += 1,
                TensorClass::TernaryCandidate { .. } => ternary += 1,
                TensorClass::Default => default += 1,
            }
        }
        (preserve, fp16, ternary, default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GROK1_TENSOR_TOTAL;

    #[test]
    fn full_inventory_has_770_tensors() {
        let inv = Grok1Inventory::full();
        assert_eq!(
            inv.len(),
            GROK1_TENSOR_TOTAL,
            "inventory must have exactly 770 tensors"
        );
    }

    #[test]
    fn class_counts_match_xai_dissect() {
        let inv = Grok1Inventory::full();
        let (preserve, fp16, ternary, default) = inv.count_by_expected_class();
        assert_eq!(
            preserve, 321,
            "321 preserve: 64 routers + 256 block_norms + 1 final_norm"
        );
        assert_eq!(
            fp16, 0,
            "no fp16 in structural manifest (embedding is ternary candidate per first-quantization-target.md)"
        );
        assert_eq!(
            ternary, 449,
            "449 ternary: 192 MoE expert + 256 attn projections + 1 token_embedding (first SAAQ target)"
        );
        assert_eq!(default, 0, "no tensors should fall to default");
    }

    #[test]
    fn no_duplicate_structural_names() {
        let inv = Grok1Inventory::full();
        let mut seen = std::collections::BTreeSet::new();
        for t in &inv.tensors {
            assert!(
                seen.insert(&t.structural_name),
                "duplicate: {}",
                t.structural_name
            );
        }
    }

    #[test]
    fn dtype_counts_match_xai_dissect() {
        let inv = Grok1Inventory::full();
        let f32_count = inv.tensors.iter().filter(|t| t.dtype == "f32").count();
        let i8_count = inv.tensors.iter().filter(|t| t.dtype == "i8").count();
        assert_eq!(f32_count, 322, "322 f32 tensors");
        assert_eq!(i8_count, 448, "448 i8 tensors");
    }

    #[test]
    fn per_block_tensor_count() {
        let inv = Grok1Inventory::full();
        let block_tensors: Vec<_> = inv.tensors.iter().filter(|t| t.block.is_some()).collect();
        assert_eq!(
            block_tensors.len(),
            768,
            "768 block tensors (64 blocks × 12)"
        );
        for blk in 0..GROK1_BLOCKS {
            let count = inv.tensors.iter().filter(|t| t.block == Some(blk)).count();
            assert_eq!(count, 12, "block {blk} should have 12 tensors, got {count}");
        }
    }
}
