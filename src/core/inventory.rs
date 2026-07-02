use crate::core::selection::TensorClass;

/// A model-agnostic inventory of tensors and their expected classification.
///
/// Implementations provide the list of tensors for a specific model (e.g., Grok-1)
/// along with methods to query classification and counts. This trait enables
/// `DryRunPlanner`, `check_alignment()`, and other generic algorithms to work
/// across multiple models without Grok-1-specific assumptions.
pub trait ModelInventory {
    /// Total number of tensors in this inventory.
    fn total_tensors(&self) -> usize;

    /// Returns a slice of all tensors in this inventory.
    fn tensors(&self) -> &[InventoryTensor];

    /// Returns counts of tensors by expected class: (preserve, fp16, ternary, default).
    fn count_by_expected_class(&self) -> (usize, usize, usize, usize);

    /// Classify a tensor by its structural name, if present in this inventory.
    fn classify_tensor(&self, structural_name: &str) -> Option<TensorClass>;

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
