//! `artifacts` subcommands: generate / validate xai-dissect reports.

use clap::Subcommand;
use grok_ozempic::reports;
use grok_ozempic::reports::schema::ArtifactIR;
use std::path::{Component, Path, PathBuf};

#[derive(Subcommand)]
pub(crate) enum ArtifactsCommands {
    /// Generate artifact reports based on a manifest or GOZ1 stream
    Generate {
        /// Path to the dissect manifest JSON file
        #[arg(long)]
        manifest: PathBuf,

        /// Output directory for the reports
        #[arg(long)]
        output_dir: PathBuf,

        /// Optional path to the raw weights directory (e.g. ckpt-0)
        /// Used to derive checkpoint provenance and observed shard count.
        #[arg(long)]
        weights_dir: Option<PathBuf>,

        /// Optional checkpoint name override. If not provided, it will be derived
        /// from the weights_dir if present, or fallback to manifest source.
        #[arg(long)]
        checkpoint: Option<String>,

        /// Optional xai-dissect `inventory.json` (schema v2). When set, IR
        /// totals are derived from that scan instead of `GROK1_*` constants.
        #[arg(long)]
        inventory: Option<PathBuf>,
    },
    /// Validate generated reports in a directory against the dissect manifest
    Validate {
        /// Directory containing generated reports to validate
        #[arg(long)]
        report_dir: PathBuf,

        /// Path to the dissect manifest JSON file (same as for `generate`)
        #[arg(long)]
        manifest: PathBuf,

        /// Optional path to the raw weights directory (e.g. ckpt-0), same semantics as `generate`
        #[arg(long)]
        weights_dir: Option<PathBuf>,

        /// Optional checkpoint name override, same semantics as `generate`
        #[arg(long)]
        checkpoint: Option<String>,

        /// Optional xai-dissect `inventory.json` (schema v2). Same semantics as `generate`.
        #[arg(long)]
        inventory: Option<PathBuf>,
    },
}

pub(crate) fn cmd_artifacts(cmd: ArtifactsCommands) -> anyhow::Result<()> {
    match cmd {
        ArtifactsCommands::Generate {
            manifest,
            output_dir,
            weights_dir,
            checkpoint,
            inventory,
        } => cmd_artifacts_generate(manifest, output_dir, weights_dir, checkpoint, inventory),
        ArtifactsCommands::Validate {
            report_dir,
            manifest,
            weights_dir,
            checkpoint,
            inventory,
        } => cmd_artifacts_validate(report_dir, manifest, weights_dir, checkpoint, inventory),
    }
}

fn cmd_artifacts_generate(
    manifest: PathBuf,
    output_dir: PathBuf,
    weights_dir: Option<PathBuf>,
    checkpoint: Option<String>,
    inventory: Option<PathBuf>,
) -> anyhow::Result<()> {
    println!(
        "Generating artifacts to {} using manifest {}",
        output_dir.display(),
        manifest.display()
    );
    let (actual_checkpoint, actual_shards) =
        resolve_checkpoint_and_shards(weights_dir.as_deref(), checkpoint, true)?;
    let ir = load_manifest_ir(
        &manifest,
        inventory.as_deref(),
        actual_checkpoint.as_deref(),
        actual_shards,
    )?;
    reports::validator::validate_ir(&ir)
        .map_err(|e| anyhow::anyhow!("Artifact validation failed: {}", e))?;
    reports::writer::write_reports(&ir, &output_dir)
        .map_err(|e| anyhow::anyhow!("Failed to write reports: {}", e))?;
    println!("Success!");
    Ok(())
}

fn cmd_artifacts_validate(
    report_dir: PathBuf,
    manifest: PathBuf,
    weights_dir: Option<PathBuf>,
    checkpoint: Option<String>,
    inventory: Option<PathBuf>,
) -> anyhow::Result<()> {
    println!(
        "Validating reports in {} using manifest {}",
        report_dir.display(),
        manifest.display()
    );
    let (actual_checkpoint, actual_shards) =
        resolve_checkpoint_and_shards(weights_dir.as_deref(), checkpoint, false)?;
    let ir = load_manifest_ir(
        &manifest,
        inventory.as_deref(),
        actual_checkpoint.as_deref(),
        actual_shards,
    )?;
    reports::writer::validate_report_dir_against_ir(&report_dir, &ir)
        .map_err(|e| anyhow::anyhow!("Artifact report validation failed: {}", e))?;
    println!("Report directory matches manifest and passes IR validation.");
    Ok(())
}

fn load_manifest_ir(
    manifest: &Path,
    inventory: Option<&Path>,
    actual_checkpoint: Option<&str>,
    actual_shards: Option<usize>,
) -> anyhow::Result<ArtifactIR> {
    let manifest_bytes = std::fs::read(manifest)?;
    let dissect_manifest = grok_ozempic::parse_manifest_bytes(
        &manifest_bytes,
        manifest.to_str().unwrap_or("manifest"),
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse manifest: {}", e))?;

    let scan = match inventory {
        Some(path) => Some(
            reports::scan::load_inventory_scan(path)
                .map_err(|e| anyhow::anyhow!("Failed to load inventory scan: {}", e))?,
        ),
        None => None,
    };

    reports::detector::build_artifact_ir(
        &dissect_manifest,
        scan.as_ref(),
        actual_checkpoint,
        actual_shards,
    )
    .map_err(|e| anyhow::anyhow!("Failed to build IR: {}", e))
}

/// Returns `(checkpoint_override, shard_count)` for [`reports::detector::build_artifact_ir`].
fn resolve_checkpoint_and_shards(
    weights_dir: Option<&Path>,
    checkpoint: Option<String>,
    log_shard_discovery: bool,
) -> anyhow::Result<(Option<String>, Option<usize>)> {
    let Some(wd) = weights_dir else {
        return Ok((checkpoint, None));
    };
    if !wd.is_dir() {
        anyhow::bail!(
            "--weights-dir is not a directory: {} (refusing silent Grok-1 default fallback)",
            wd.display()
        );
    }
    let checkpoint = checkpoint.or_else(|| derive_checkpoint_name_from_weights_dir(wd));
    let actual_shards = discover_xai_tensor_shards(wd, log_shard_discovery)?;
    Ok((checkpoint, actual_shards))
}

fn derive_checkpoint_name_from_weights_dir(wd: &Path) -> Option<String> {
    let mut tail: Vec<_> = wd
        .components()
        .rev()
        .filter_map(|c| match c {
            Component::Normal(name) => Some(name),
            _ => None,
        })
        .take(2)
        .collect();
    tail.reverse();
    match tail.as_slice() {
        [only] => Some(only.to_string_lossy().to_string()),
        [parent, leaf] => Some(format!(
            "{}/{}",
            parent.to_string_lossy(),
            leaf.to_string_lossy()
        )),
        _ => None,
    }
}

fn discover_xai_tensor_shards(
    wd: &Path,
    log_shard_discovery: bool,
) -> anyhow::Result<Option<usize>> {
    let count = count_xai_tensor_shards(wd)?;
    if count == 0 {
        anyhow::bail!(
            "--weights-dir {} contains no xai-dissect tensor*_* shards (refusing silent Grok-1 default fallback)",
            wd.display()
        );
    }
    if log_shard_discovery {
        println!("Discovered {} xai-dissect tensor shards.", count);
    }
    Ok(Some(count))
}

fn count_xai_tensor_shards(dir: &Path) -> anyhow::Result<usize> {
    let mut count = 0usize;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.path().is_file() {
            continue;
        }
        if entry
            .file_name()
            .to_str()
            .is_some_and(is_xai_tensor_shard_name)
        {
            count += 1;
        }
    }
    Ok(count)
}

fn is_xai_tensor_shard_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("tensor") else {
        return false;
    };
    let Some((major, minor)) = rest.split_once('_') else {
        return false;
    };
    major.len() == 5
        && minor.len() == 3
        && major.bytes().all(|byte| byte.is_ascii_digit())
        && minor.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use grok_ozempic::GROK1_BASELINE_JSON;
    use grok_ozempic::core::stream::GROK1_BLOCK_COUNT;
    use grok_ozempic::types::{
        GROK1_BLOCK_SLOTS, GROK1_HIDDEN_DIM, GROK1_TENSOR_F32, GROK1_TENSOR_INT8,
        GROK1_TENSOR_QUANT, GROK1_TENSOR_TOTAL, GROK1_TENSOR_TOTAL_BYTES,
        GROK1_TENSOR_TOTAL_ELEMENTS, GROK1_VOCAB_SIZE,
    };
    use std::fs;
    use std::sync::OnceLock;

    fn fixture_dir() -> PathBuf {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("artifacts-cli-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_baseline_manifest(dir: &Path) -> PathBuf {
        let path = dir.join("baseline.json");
        fs::write(&path, GROK1_BASELINE_JSON).unwrap();
        path
    }

    fn slot_json(slot: &grok_ozempic::types::BlockSlot) -> String {
        let (role, dtype) = if slot.is_int8 {
            ("quant_weight", "i8")
        } else {
            ("tensor", "f32")
        };
        let shape = slot
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            r#"{{"role":"{role}","dtype":"{dtype}","shape":[{shape}],"nbytes":{}}}"#,
            slot.bytes
        )
    }

    fn grok1_inventory_json() -> &'static str {
        static JSON: OnceLock<String> = OnceLock::new();
        JSON.get_or_init(|| {
            let mut tensors = Vec::with_capacity(GROK1_TENSOR_TOTAL);
            tensors.push(format!(
                r#"{{"role":"tensor","dtype":"f32","shape":[{}, {}],"nbytes":{}}}"#,
                GROK1_VOCAB_SIZE,
                GROK1_HIDDEN_DIM,
                GROK1_VOCAB_SIZE * GROK1_HIDDEN_DIM * 4
            ));
            for _ in 0..GROK1_BLOCK_COUNT {
                tensors.extend(GROK1_BLOCK_SLOTS.iter().map(slot_json));
            }
            tensors.push(format!(
                r#"{{"role":"tensor","dtype":"f32","shape":[{}],"nbytes":{}}}"#,
                GROK1_HIDDEN_DIM,
                GROK1_HIDDEN_DIM * 4
            ));
            format!(
                r#"{{"model_family":"grok-1","checkpoint_path":"grok-1-official/ckpt-0","shard_count":{GROK1_TENSOR_TOTAL},"tensors":[{}],"totals":{{"tensors":{GROK1_TENSOR_TOTAL},"quant_tensors":{GROK1_TENSOR_QUANT},"f32_tensors":{GROK1_TENSOR_F32},"i8_tensors":{GROK1_TENSOR_INT8},"total_nbytes":{GROK1_TENSOR_TOTAL_BYTES},"total_elements":{GROK1_TENSOR_TOTAL_ELEMENTS}}},"schema_version":2}}"#,
                tensors.join(",")
            )
        })
    }

    fn mini_inventory_json() -> &'static str {
        r#"{"model_family":"grok-1","checkpoint_path":"/fixtures/ckpt-0","shard_count":2,"tensors":[{"role":"tensor","dtype":"f32","shape":[4],"nbytes":16},{"role":"quant_weight","dtype":"i8","shape":[5],"nbytes":5}],"totals":{"tensors":2,"quant_tensors":1,"f32_tensors":1,"i8_tensors":1,"total_nbytes":21,"total_elements":9},"schema_version":2}"#
    }

    #[test]
    fn load_manifest_ir_spec_path_without_inventory() {
        let dir = fixture_dir();
        let manifest = write_baseline_manifest(&dir);
        let ir = load_manifest_ir(&manifest, None, None, None).expect("spec IR");
        assert_eq!(ir.totals.total, GROK1_TENSOR_TOTAL);
        assert_eq!(ir.manifest.shards, GROK1_TENSOR_TOTAL);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_manifest_ir_derives_totals_from_inventory() {
        let dir = fixture_dir();
        let manifest = write_baseline_manifest(&dir);
        let inventory = dir.join("inventory.json");
        fs::write(&inventory, grok1_inventory_json()).unwrap();
        let ir = load_manifest_ir(&manifest, Some(&inventory), Some("cli-ckpt"), None)
            .expect("scan-backed IR");
        assert_eq!(ir.totals.total, GROK1_TENSOR_TOTAL);
        assert_eq!(ir.manifest.checkpoint, "cli-ckpt");
        assert_eq!(ir.manifest.shards, GROK1_TENSOR_TOTAL);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_manifest_ir_rejects_missing_inventory() {
        let dir = fixture_dir();
        let manifest = write_baseline_manifest(&dir);
        let err = load_manifest_ir(&manifest, Some(&dir.join("missing.json")), None, None)
            .expect_err("missing inventory");
        assert!(
            err.to_string().contains("Failed to load inventory scan"),
            "got {err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_manifest_ir_rejects_bad_inventory_json() {
        let dir = fixture_dir();
        let manifest = write_baseline_manifest(&dir);
        let inventory = dir.join("inventory.json");
        fs::write(&inventory, "{ not json").unwrap();
        let err = load_manifest_ir(&manifest, Some(&inventory), None, None)
            .expect_err("malformed inventory");
        assert!(
            err.to_string().contains("Failed to load inventory scan"),
            "got {err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cmd_artifacts_generate_and_validate_without_inventory() {
        let dir = fixture_dir();
        let manifest = write_baseline_manifest(&dir);
        let out = dir.join("reports");
        cmd_artifacts(ArtifactsCommands::Generate {
            manifest: manifest.clone(),
            output_dir: out.clone(),
            weights_dir: None,
            checkpoint: Some("from-cli".into()),
            inventory: None,
        })
        .expect("generate spec reports");
        assert!(out.join("inventory.md").is_file());
        cmd_artifacts(ArtifactsCommands::Validate {
            report_dir: out,
            manifest,
            weights_dir: None,
            checkpoint: Some("from-cli".into()),
            inventory: None,
        })
        .expect("validate spec reports");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cmd_artifacts_generate_and_validate_with_inventory() {
        let dir = fixture_dir();
        let manifest = write_baseline_manifest(&dir);
        let inventory = dir.join("inventory.json");
        fs::write(&inventory, grok1_inventory_json()).unwrap();
        let out = dir.join("reports");
        cmd_artifacts(ArtifactsCommands::Generate {
            manifest: manifest.clone(),
            output_dir: out.clone(),
            weights_dir: None,
            checkpoint: None,
            inventory: Some(inventory.clone()),
        })
        .expect("generate scan-backed reports");
        cmd_artifacts(ArtifactsCommands::Validate {
            report_dir: out,
            manifest,
            weights_dir: None,
            checkpoint: None,
            inventory: Some(inventory),
        })
        .expect("validate scan-backed reports");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cmd_artifacts_generate_rejects_non_grok1_inventory_totals() {
        let dir = fixture_dir();
        let manifest = write_baseline_manifest(&dir);
        let inventory = dir.join("inventory.json");
        fs::write(&inventory, mini_inventory_json()).unwrap();
        let err = cmd_artifacts(ArtifactsCommands::Generate {
            manifest,
            output_dir: dir.join("reports"),
            weights_dir: None,
            checkpoint: None,
            inventory: Some(inventory),
        })
        .expect_err("mini scan must fail IR validation");
        assert!(
            err.to_string().contains("Artifact validation failed"),
            "got {err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cmd_artifacts_generate_rejects_weights_dir_inventory_shard_mismatch() {
        let dir = fixture_dir();
        let manifest = write_baseline_manifest(&dir);
        let inventory = dir.join("inventory.json");
        fs::write(&inventory, grok1_inventory_json()).unwrap();
        let weights = dir.join("ckpt-0");
        fs::create_dir_all(&weights).unwrap();
        fs::write(weights.join("tensor00000_000"), b"x").unwrap();
        let err = cmd_artifacts(ArtifactsCommands::Generate {
            manifest,
            output_dir: dir.join("reports"),
            weights_dir: Some(weights),
            checkpoint: None,
            inventory: Some(inventory),
        })
        .expect_err("1 discovered shard vs 770 in the scan");
        assert!(
            err.to_string()
                .contains("does not match inventory scan shard_count"),
            "got {err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shard_name_helper_accepts_xai_dissect_pattern() {
        assert!(is_xai_tensor_shard_name("tensor00000_000"));
        assert!(!is_xai_tensor_shard_name("tensor0_0"));
        assert!(!is_xai_tensor_shard_name("weights.bin"));
    }
}
