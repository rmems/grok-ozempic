//! SAAQ-derived per-tensor GIF threshold source.
//!
//! `corinth-canal` emits a small JSON map (`saaq-tau-map` schema) translating
//! its `saaq_delta_q_target` telemetry signal into per-tensor/tier
//! `gif_threshold` multipliers. This module loads that map so
//! `quantize-goz1` can apply activity-derived thresholds through the normal
//! [`crate::core::precision::decide_for_tensor`] path — the applied value
//! still lands in the GOZ1 v3 tensor row (`gif_threshold` + `threshold_abs`),
//! so `oz.gif_threshold` semantics and the #58/#66 reporting story are
//! unchanged.
//!
//! ## File format (`corinth-canal/saaq-tau-map`, version 1)
//!
//! ```json
//! {
//!   "schema": "corinth-canal/saaq-tau-map",
//!   "version": 1,
//!   "entries": [
//!     {
//!       "pattern": "block_*.slot_00.moe_expert.gate",
//!       "tier": "moe_expert",
//!       "gif_threshold": 0.65,
//!       "saaq_delta_q_target": 0.5
//!     }
//!   ]
//! }
//! ```
//!
//! `entries[].pattern` uses the same segment-anchored glob vocabulary as
//! manifest `ternary_candidates[].name`
//! ([`crate::core::selection::glob_match`]); entries are consulted in file
//! order and the first match wins. `tier` / `saaq_delta_q_target` are
//! provenance carried through for reporting; only `pattern` and
//! `gif_threshold` are load-bearing.
//!
//! Threshold resolution order (see [`crate::core::precision`]):
//!
//! 1. Per-tensor `ternary_candidates[].gif_threshold` (explicit manifest
//!    authoring always wins — a SAAQ map cannot silently override it).
//! 2. This map, looked up by tensor name.
//! 3. `manifest.defaults.gif_threshold`.
//! 4. `config.gif_threshold` (CLI `--gif-threshold`).
//!
//! ## Dry-run caveat
//!
//! [`crate::core::dry_run::DryRunPlanner`] resolves SAAQ values per manifest
//! *rule name*: a SAAQ pattern identical to (or glob-matching) a
//! `ternary_candidates` rule name shows up in the plan row. A SAAQ map that
//! subdivides one manifest rule into several τ values still applies correctly
//! at pack time — the v3 tensor row is authoritative — but the plan reports
//! the rule-level resolution only.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    core::selection::glob_match,
    error::{GrokOzempicError, Result},
    types::validate_gif_threshold,
};

/// One entry of a [`SaaqTauMap`]: a name pattern and the `gif_threshold`
/// multiplier a SAAQ control loop derived for the tensors it matches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SaaqTauEntry {
    /// Exact tensor name or segment-glob pattern (same vocabulary as manifest
    /// `ternary_candidates[].name`), e.g. `block_*.slot_00.moe_expert.gate`.
    pub pattern: String,
    /// SAAQ-derived `gif_threshold` multiplier (`τ = gif_threshold × rms`).
    pub gif_threshold: f32,
    /// Optional tier label for reporting (`moe_expert`, `attention`, ...).
    #[serde(default)]
    pub tier: Option<String>,
    /// The `saaq_delta_q_target` this τ was derived from (provenance only).
    #[serde(default)]
    pub saaq_delta_q_target: Option<f32>,
}

/// A parsed `saaq-tau-map` file.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SaaqTauMap {
    /// Producer-declared schema tag (`corinth-canal/saaq-tau-map`).
    /// Informational: any JSON with an `entries` list is accepted so a
    /// hand-authored or synthetic map works identically.
    #[serde(default)]
    pub schema: Option<String>,
    /// Schema version; informational for the same reason.
    #[serde(default)]
    pub version: Option<u32>,
    /// Pattern → threshold entries in file order; first match wins.
    #[serde(default)]
    pub entries: Vec<SaaqTauEntry>,
}

impl SaaqTauMap {
    /// Resolve the SAAQ-derived `gif_threshold` for a tensor name.
    ///
    /// Entries are tried in order; the first whose `pattern` glob-matches
    /// `tensor_name` wins. Returns `None` when nothing matches.
    pub fn lookup(&self, tensor_name: &str) -> Option<f32> {
        self.entries
            .iter()
            .find(|e| glob_match(&e.pattern, tensor_name))
            .map(|e| e.gif_threshold)
    }
}

/// Load and validate a `saaq-tau-map` JSON file.
///
/// Every entry's `gif_threshold` is run through
/// [`validate_gif_threshold`] exactly like the manifest/CLI path — a
/// non-finite or negative SAAQ value is rejected at load time rather than
/// reaching the quantizer.
pub fn load_saaq_tau_map(path: &Path) -> Result<SaaqTauMap> {
    let bytes = std::fs::read(path).map_err(|source| GrokOzempicError::ManifestIo {
        path: path.display().to_string(),
        source,
    })?;
    let map: SaaqTauMap =
        serde_json::from_slice(&bytes).map_err(|source| GrokOzempicError::ManifestParse {
            path: path.display().to_string(),
            source,
        })?;
    for entry in &map.entries {
        validate_gif_threshold(entry.gif_threshold).map_err(|msg| {
            GrokOzempicError::InvalidConfig(format!(
                "saaq-tau-map {} entry {:?}: {}",
                path.display(),
                entry.pattern,
                msg
            ))
        })?;
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn map_with(entries: Vec<SaaqTauEntry>) -> SaaqTauMap {
        SaaqTauMap {
            schema: Some("corinth-canal/saaq-tau-map".into()),
            version: Some(1),
            entries,
        }
    }

    #[test]
    fn lookup_matches_exact_and_glob_patterns_in_order() {
        let map = map_with(vec![
            SaaqTauEntry {
                pattern: "block_*.slot_00.moe_expert.gate".into(),
                gif_threshold: 0.65,
                tier: Some("moe_expert".into()),
                saaq_delta_q_target: Some(0.5),
            },
            SaaqTauEntry {
                pattern: "block_000.slot_00.moe_expert.gate".into(),
                gif_threshold: 0.9,
                tier: None,
                saaq_delta_q_target: None,
            },
        ]);
        // First match wins: the tier glob precedes the exact name.
        assert_eq!(map.lookup("block_000.slot_00.moe_expert.gate"), Some(0.65));
        assert_eq!(map.lookup("block_001.slot_00.moe_expert.gate"), Some(0.65));
        assert_eq!(map.lookup("block_000.slot_11.router"), None);
    }

    #[test]
    fn lookup_supports_segment_globs_only() {
        // Same vocabulary as manifest rules: `*` matches within a segment,
        // segment counts must agree.
        let map = map_with(vec![SaaqTauEntry {
            pattern: "block_*.slot_01.moe_expert.down".into(),
            gif_threshold: 0.8,
            tier: None,
            saaq_delta_q_target: None,
        }]);
        assert_eq!(map.lookup("block_042.slot_01.moe_expert.down"), Some(0.8));
        assert_eq!(map.lookup("block_042.slot_01.extra.moe_expert.down"), None);
    }

    // Prefer crate target/ over shared OS temp_dir (Semgrep temp-dir),
    // matching the quantize.rs test helper.
    fn test_dir() -> std::path::PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("saaq-map-tests")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn load_rejects_negative_threshold() {
        // NaN/Infinity are not JSON literals, so serde_json rejects them at
        // parse time; a negative value parses fine and must be caught by
        // validate_gif_threshold instead.
        let path = test_dir().join("bad.json");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(
            f,
            r#"{{"entries": [{{"pattern": "a.b", "gif_threshold": -0.1}}]}}"#
        )
        .unwrap();
        drop(f);
        let err = load_saaq_tau_map(&path).unwrap_err();
        assert!(
            matches!(err, GrokOzempicError::InvalidConfig(_)),
            "negative value should be InvalidConfig, got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_round_trips_minimal_map() {
        let path = test_dir().join("ok.json");
        std::fs::write(
            &path,
            r#"{
                "schema": "corinth-canal/saaq-tau-map",
                "version": 1,
                "entries": [
                    {"pattern": "embedding.slot_00.token_embedding", "gif_threshold": 0.65}
                ]
            }"#,
        )
        .unwrap();
        let map = load_saaq_tau_map(&path).unwrap();
        assert_eq!(map.entries.len(), 1);
        assert_eq!(map.lookup("embedding.slot_00.token_embedding"), Some(0.65));
        let _ = std::fs::remove_file(&path);
    }
}
