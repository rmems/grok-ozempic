use crate::core::grok1_data::build_grok1_tensors;
use crate::core::inventory::{InventoryTensor, ModelInventory};
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
}

impl ModelInventory for Grok1Inventory {
    fn total_tensors(&self) -> usize {
        self.tensors.len()
    }

    fn tensors(&self) -> &[InventoryTensor] {
        &self.tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::grok1_data::GROK1_BLOCKS;
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

    /// The shared 12-slot table and the three tables that used to duplicate it
    /// must agree (GH #106).
    ///
    /// Nothing previously asserted that, which is exactly why they drifted:
    /// slots 00 and 02 read `moe_expert.unresolved` in two copies while the
    /// canonical manifest and this inventory had resolved them to `.gate` and
    /// `.up`. This test fails if any single table is edited alone.
    #[test]
    fn block_slot_table_matches_the_core_inventory() {
        use crate::types::GROK1_BLOCK_SLOTS;

        assert_eq!(GROK1_BLOCK_SLOTS.len(), 12, "a Grok-1 block has 12 slots");

        // Slot indices are 0..=11 exactly once, in order.
        for (i, s) in GROK1_BLOCK_SLOTS.iter().enumerate() {
            assert_eq!(s.slot, i, "slot table must be dense and ordered");
        }

        // Tier split: 8 int8 (3 expert + 4 attention... plus slot 0/1/2) vs 5 f32.
        let int8 = GROK1_BLOCK_SLOTS.iter().filter(|s| s.is_int8).count();
        let preserve = GROK1_BLOCK_SLOTS.iter().filter(|s| s.is_preserve()).count();
        assert_eq!((int8, preserve), (7, 5), "7 int8 + 5 f32 per block");

        // The resolved MoE labels, matching dissect/grok-1/structural-manifest.json.
        // `moe_expert.unresolved` must not reappear.
        let kinds: Vec<&str> = GROK1_BLOCK_SLOTS.iter().map(|s| s.kind).collect();
        assert_eq!(kinds[0], "moe_expert.gate");
        assert_eq!(kinds[1], "moe_expert.down");
        assert_eq!(kinds[2], "moe_expert.up");
        assert!(
            !kinds.iter().any(|k| k.contains("unresolved")),
            "slots 00/02 are resolved to .gate/.up; 'unresolved' is stale (GH #106)"
        );
        assert_eq!(kinds[11], "router");

        // Every kind the core inventory emits for a block must exist in the
        // table, and vice versa -- this is the cross-table link that was missing.
        let table_kinds: std::collections::BTreeSet<&str> = kinds.iter().copied().collect();
        let inventory_kinds: std::collections::BTreeSet<String> = build_grok1_tensors()
            .into_iter()
            .filter(|t| t.block.is_some())
            .map(|t| t.kind.to_string())
            .collect();
        let inventory_refs: std::collections::BTreeSet<&str> =
            inventory_kinds.iter().map(String::as_str).collect();
        assert_eq!(
            table_kinds, inventory_refs,
            "GROK1_BLOCK_SLOTS kinds must match the per-block kinds grok1_full_inventory emits"
        );

        // dtype spellings stay distinct on purpose (artifact.index.json says
        // "int8", the core inventory says "i8"); both come from one flag.
        let expert = &GROK1_BLOCK_SLOTS[0];
        assert_eq!(expert.dtype_artifact(), "int8");
        assert_eq!(expert.dtype_inventory(), "i8");
        assert_eq!(GROK1_BLOCK_SLOTS[11].dtype_artifact(), "f32");
        assert_eq!(GROK1_BLOCK_SLOTS[11].dtype_inventory(), "f32");
    }
}
