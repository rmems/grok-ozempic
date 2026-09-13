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
    fn block_slot_table_is_dense_and_correctly_tiered() {
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
    }

    /// The shared table must agree with the core inventory slot-for-slot.
    #[test]
    fn block_slot_table_matches_the_core_inventory() {
        use crate::types::GROK1_BLOCK_SLOTS;

        // Cross-table link: compare (slot, kind, dtype) PER SLOT, not as sets.
        //
        // A set comparison would have been useless here, which is worth spelling
        // out because the first version of this test did exactly that. Collapsing
        // both sides to a `BTreeSet<&str>` of kinds discards two things the drift
        // in GH #106 actually consisted of:
        //
        //   - slot position: `moe_expert.gate` and `moe_expert.up` have identical
        //     shapes, so swapping them between slots 00 and 02 leaves the kind set
        //     unchanged while every structural name points at the wrong tensor.
        //   - multiplicity: the four `block_norm` slots collapse to one entry, so
        //     dropping three of them would still compare equal.
        //
        // Comparing the full per-slot record catches both.
        let mut inventory_by_slot: Vec<(u32, &str, &str)> = build_grok1_tensors()
            .iter()
            .filter(|t| t.block == Some(0))
            .map(|t| (t.slot.expect("block tensors carry a slot"), t.kind, t.dtype))
            .collect();
        inventory_by_slot.sort_by_key(|(slot, _, _)| *slot);

        let table_by_slot: Vec<(u32, &str, &str)> = GROK1_BLOCK_SLOTS
            .iter()
            .map(|s| (s.slot as u32, s.kind, s.dtype_inventory()))
            .collect();

        assert_eq!(
            table_by_slot, inventory_by_slot,
            "GROK1_BLOCK_SLOTS must match build_grok1_tensors() slot-for-slot \
             (slot, kind, dtype) -- a set comparison would miss a gate/up swap \
             or a dropped duplicate block_norm"
        );
    }

    /// Every block must share the same slot layout, so a per-block special
    /// case cannot hide behind block 0 being correct.
    #[test]
    fn every_block_shares_the_slot_layout() {
        use crate::types::GROK1_BLOCK_SLOTS;

        let table_by_slot: Vec<(u32, &str, &str)> = GROK1_BLOCK_SLOTS
            .iter()
            .map(|s| (s.slot as u32, s.kind, s.dtype_inventory()))
            .collect();

        // Same check across every block, so a per-block special case cannot hide.
        for blk in [1u32, 31, 63] {
            let mut per_block: Vec<(u32, &str, &str)> = build_grok1_tensors()
                .iter()
                .filter(|t| t.block == Some(blk))
                .map(|t| (t.slot.expect("slot"), t.kind, t.dtype))
                .collect();
            per_block.sort_by_key(|(slot, _, _)| *slot);
            assert_eq!(
                per_block, table_by_slot,
                "block {blk} must have the same slot layout as the shared table"
            );
        }

        // dtype spellings stay distinct on purpose (artifact.index.json says
        // "int8", the core inventory says "i8"); both come from one flag.
        let expert = &GROK1_BLOCK_SLOTS[0];
        assert_eq!(expert.dtype_artifact(), "int8");
        assert_eq!(expert.dtype_inventory(), "i8");
        assert_eq!(GROK1_BLOCK_SLOTS[11].dtype_artifact(), "f32");
        assert_eq!(GROK1_BLOCK_SLOTS[11].dtype_inventory(), "f32");
    }
}
