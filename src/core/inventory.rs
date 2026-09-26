use crate::core::selection::TensorClass;

/// A model-agnostic inventory of tensors and their expected classification.
///
/// Implementations provide the list of tensors for a specific model family
/// along with methods to query classification and counts. This trait enables
/// `DryRunPlanner`, `check_alignment()`, and other generic algorithms to work
/// across multiple models without baking a single family's layout into the
/// engine. Grok-1 is the reference [`crate::core::models::grok1::Grok1Profile`];
/// [`VecInventory`] is the escape hatch for fixtures and new families.
pub trait ModelInventory {
    /// Total number of tensors in this inventory.
    fn total_tensors(&self) -> usize;

    /// Returns a slice of all tensors in this inventory.
    fn tensors(&self) -> &[InventoryTensor];

    /// Count how many tensors in this inventory match the given glob pattern.
    /// Default implementation iterates and uses `glob_match`.
    fn count_matching(&self, pattern: &str) -> usize {
        self.tensors()
            .iter()
            .filter(|t| crate::core::selection::glob_match(pattern, &t.structural_name))
            .count()
    }

    /// Default: derive counts by iterating `tensors()`.
    fn count_by_expected_class(&self) -> (usize, usize, usize, usize) {
        let mut preserve = 0;
        let mut fp16 = 0;
        let mut ternary = 0;
        let mut default = 0;
        for t in self.tensors() {
            match &t.expected_class {
                TensorClass::Preserve { .. } => preserve += 1,
                TensorClass::Fp16 { .. } => fp16 += 1,
                TensorClass::TernaryCandidate { .. } => ternary += 1,
                TensorClass::Default => default += 1,
            }
        }
        (preserve, fp16, ternary, default)
    }

    /// Default: derive classification by iterating `tensors()`.
    fn classify_tensor(&self, structural_name: &str) -> Option<TensorClass> {
        self.tensors()
            .iter()
            .find(|t| t.structural_name == structural_name)
            .map(|t| t.expected_class.clone())
    }
}

/// A single tensor entry in a model inventory.
#[derive(Debug, Clone, PartialEq)]
pub struct InventoryTensor {
    pub structural_name: String,
    pub expected_class: TensorClass,
    pub dtype: &'static str,
    pub block: Option<u32>,
    pub slot: Option<u32>,
    pub kind: &'static str,
}

/// Inventory backed by an owned tensor list.
///
/// Use this for tests and for small in-tree plugins (see
/// [`crate::core::models::toy_moe`]) so a new family does not need a
/// dedicated struct just to implement [`ModelInventory`].
#[derive(Debug, Clone, PartialEq)]
pub struct VecInventory {
    tensors: Vec<InventoryTensor>,
}

impl VecInventory {
    /// Wrap an owned tensor list.
    pub fn new(tensors: Vec<InventoryTensor>) -> Self {
        Self { tensors }
    }
}

impl From<Vec<InventoryTensor>> for VecInventory {
    fn from(tensors: Vec<InventoryTensor>) -> Self {
        Self::new(tensors)
    }
}

impl ModelInventory for VecInventory {
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

    fn tensor(name: &str, class: TensorClass) -> InventoryTensor {
        InventoryTensor {
            structural_name: name.into(),
            expected_class: class,
            dtype: "f32",
            block: None,
            slot: None,
            kind: "fixture",
        }
    }

    #[test]
    fn from_vec_classifies_and_counts() {
        let inv = VecInventory::from(vec![
            tensor("keep", TensorClass::Preserve { reason: None }),
            tensor("half", TensorClass::Fp16 { reason: None }),
            tensor(
                "tri",
                TensorClass::TernaryCandidate {
                    rank: None,
                    gif_threshold: None,
                },
            ),
            tensor("other", TensorClass::Default),
        ]);
        assert_eq!(inv.total_tensors(), 4);
        assert_eq!(
            inv.classify_tensor("half"),
            Some(TensorClass::Fp16 { reason: None })
        );
        assert_eq!(inv.classify_tensor("missing"), None);
        assert_eq!(inv.count_matching("half"), 1);
        assert_eq!(inv.count_by_expected_class(), (1, 1, 1, 1));
    }
}
