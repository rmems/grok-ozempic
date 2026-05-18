pub mod detector;
pub mod schema;
pub mod templates;
pub mod validator;
pub mod writer;

/// Canonical Grok-1 expert slot shape strings for IR generation and validation.
pub(crate) fn grok1_expected_expert_shape_strings() -> [String; 3] {
    use crate::core::stream::{GROK1_EXPERT_COUNT, GROK1_FEED_FORWARD_LENGTH};
    use crate::types::GROK1_HIDDEN_DIM;

    [
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
    ]
}

#[cfg(test)]
mod tests;
