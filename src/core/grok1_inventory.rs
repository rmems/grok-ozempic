use crate::core::grok1_data::build_grok1_tensors;
use crate::core::inventory::{InventoryTensor, ModelInventory};
use crate::core::selection::{TensorClass, glob_match};
use crate::core::stream::GROK1_BLOCK_COUNT;

pub const GROK1_BLOCKS: u32 = GROK1_BLOCK_COUNT;
pub const GROK1_NORMS_PER_BLOCK: usize = 4;
pub const GROK1_EXPERT_PROJECTIONS_PER_BLOCK: usize = 3;
pub const GROK1_ATTN_PROJECTIONS_PER_BLOCK: usize = 4;
pub const GROK1_ROUTERS_PER_BLOCK: usize = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct Grok1Inventory {
    pub tensors: Vec<InventoryTensor>,
}

impl Grok1Inventory {
    pub fn full() -> Self {
        let tensors = build_grok1_tensors();
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

    /// Count how many tensors in the full Grok-1 inventory match the given
    /// (structural) glob pattern using the same `glob_match` logic as
    /// classification. This gives exact counts (e.g. 64 for `block_*.slot_11.router`)
    /// instead of the legacy `estimate_tensor_count` heuristic.
    ///
    /// Intended for DryRunPlanner (and similar) when operating against the
    /// xai-dissect structural (V2) manifest so that coverage summaries are
    /// accurate for the 770-tensor inventory.
    pub fn count_matching_glob(&self, pattern: &str) -> usize {
        self.tensors
            .iter()
            .filter(|t| glob_match(pattern, &t.structural_name))
            .count()
    }
}

impl ModelInventory for Grok1Inventory {
    fn total_tensors(&self) -> usize {
        self.tensors.len()
    }

    fn tensors(&self) -> &[InventoryTensor] {
        &self.tensors
    }

    fn count_by_expected_class(&self) -> (usize, usize, usize, usize) {
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

    fn classify_tensor(&self, structural_name: &str) -> Option<TensorClass> {
        self.tensors
            .iter()
            .find(|t| t.structural_name == structural_name)
            .map(|t| t.expected_class.clone())
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
