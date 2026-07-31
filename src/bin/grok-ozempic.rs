use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use grok_ozempic::artifact::{self, ConvertOptions, GROK1_ARTIFACT_FORMAT, SmokeOptions};
use grok_ozempic::reports;
use grok_ozempic::reports::schema::ArtifactIR;
use grok_ozempic::types::{
    QuantizationConfig, QuantizationInputFormat, quantize_goz1_config, validate_gif_threshold,
};
use grok_ozempic::{ShardStats, run_quantization, verify_pack_file};
use std::path::{Component, Path, PathBuf};

#[derive(Parser)]
#[command(name = "grok-ozempic")]
#[command(about = "SNN-logic quantization for Grok models", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate and validate xai-dissect compatible artifacts
    Artifacts {
        #[command(subcommand)]
        cmd: ArtifactsCommands,
    },
    /// Validate ingest-time xai-dissect manifest/checksum inputs
    ValidateIngest {
        /// Path to the xai-dissect manifest JSON file
        #[arg(long)]
        manifest: PathBuf,

        /// Optional checkpoint directory; if it contains checksums.json entries are enforced
        #[arg(long)]
        checkpoint: Option<PathBuf>,
    },
    /// Convert a Grok-1 manifest/checkpoint into deterministic saaq-g1-v0 metadata artifacts
    ConvertGrok1 {
        /// Optional checkpoint directory; dry-run mode can omit large checkpoint payloads
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Path to the xai-dissect Grok-1 conversion manifest
        #[arg(long)]
        manifest: PathBuf,

        /// Output directory for manifest.used.json, artifact.index.json, checksums, warnings, summary
        #[arg(long)]
        output_root: PathBuf,

        /// Output artifact format; only saaq-g1-v0 is supported in this sprint
        #[arg(long, default_value = GROK1_ARTIFACT_FORMAT)]
        format: String,

        /// Protect routers as f32 pass-through tensors
        #[arg(long, action = ArgAction::Set, default_value_t = true)]
        protect_routers: bool,

        /// Protect block norms and final norm as f32 pass-through tensors
        #[arg(long, action = ArgAction::Set, default_value_t = true)]
        protect_norms: bool,

        /// Validate and write metadata reports without creating large payload artifacts
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Validate/convert a one-block Grok-1 smoke slice
    SmokeGrok1 {
        /// Optional checkpoint directory; dry-run mode can omit large checkpoint payloads
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Path to the xai-dissect Grok-1 conversion manifest
        #[arg(long)]
        manifest: PathBuf,

        /// Block index to select, e.g. 0 for block_000
        #[arg(long, default_value_t = 0)]
        block: usize,

        /// Include embedding.slot_00.token_embedding in the smoke slice
        #[arg(long, action = ArgAction::Set, default_value_t = true)]
        include_embedding: bool,

        /// Include final_norm.slot_00.final_norm in the smoke slice
        #[arg(long, action = ArgAction::Set, default_value_t = true)]
        include_final_norm: bool,

        /// Output directory for smoke summary/index/checksum/warnings files
        #[arg(long)]
        output_root: PathBuf,

        /// Validate and write metadata reports without creating large payload artifacts
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Validate a full Grok-1 saaq-g1-v0 artifact index against the xai-dissect manifest
    ValidateGrok1Artifact {
        /// Path to the original xai-dissect Grok-1 manifest
        #[arg(long)]
        manifest: PathBuf,

        /// Path to artifact.index.json emitted by convert-grok1
        #[arg(long)]
        artifact_index: PathBuf,

        /// Optional checksums.json emitted by convert-grok1
        #[arg(long)]
        checksums: Option<PathBuf>,

        /// Output directory for validation.summary/report/failures/warnings
        #[arg(long)]
        output_root: Option<PathBuf>,

        /// Require every router to be protected/pass-through/f32
        #[arg(long, action = ArgAction::Set, default_value_t = true)]
        strict_router_protection: bool,
    },
    /// Pack safetensors or `.npy` weights into a GOZ1 checkpoint (real quantize path)
    ///
    /// Unlike `convert-grok1` / `smoke-grok1` (metadata-only), this streams tensor
    /// payloads through `run_quantization`. Input must not be official pickle shards —
    /// export to `.npy` first (see scripts/export_grok1_embedding_npy.py, #37 / RM-189).
    QuantizeGoz1 {
        /// Directory of `*.safetensors` shards or flat `*.npy` files
        #[arg(long)]
        input_dir: PathBuf,

        /// Output GOZ1 packed checkpoint path
        #[arg(long)]
        output: PathBuf,

        /// Optional xai-dissect manifest JSON (V1 names for runtime quantize)
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// Input layout: npy (default, JAX export) or safetensors
        #[arg(long, value_enum, default_value_t = CliInputFormat::Npy)]
        input_format: CliInputFormat,

        /// GIF saliency threshold (default 0.05; manifest defaults may override)
        #[arg(long)]
        gif_threshold: Option<f32>,

        /// Use embedded Grok-1 baseline when `--manifest` is omitted
        #[arg(long, default_value_t = false)]
        use_embedded_baseline: bool,

        /// Verify GOZ1 container after write
        #[arg(long, default_value_t = false)]
        verify: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliInputFormat {
    Safetensors,
    Npy,
}

impl From<CliInputFormat> for QuantizationInputFormat {
    fn from(v: CliInputFormat) -> Self {
        match v {
            CliInputFormat::Safetensors => QuantizationInputFormat::Safetensors,
            CliInputFormat::Npy => QuantizationInputFormat::NpyDir,
        }
    }
}

#[derive(Subcommand)]
enum ArtifactsCommands {
    /// Generate artifact reports based on a manifest or GOZ1 stream
    Generate {
        /// Path to the dissect manifest JSON file
        #[arg(long)]
        manifest: PathBuf,

        /// Output directory for the reports
        #[arg(long)]
        output_dir: PathBuf,

        /// Optional path to the raw weights directory (e.g. ckpt-0)
        /// Used to derive real checkpoint provenance and tensor totals.
        #[arg(long)]
        weights_dir: Option<PathBuf>,

        /// Optional checkpoint name override. If not provided, it will be derived
        /// from the weights_dir if present, or fallback to manifest source.
        #[arg(long)]
        checkpoint: Option<String>,
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
    },
}

fn main() -> anyhow::Result<()> {
    run_cli(Cli::parse().command)
}

// Keep the dispatcher compact so Lizard NLOC stays under Codacy's threshold.
#[rustfmt::skip]
fn run_cli(command: Commands) -> anyhow::Result<()> {
    match command {
        Commands::ValidateIngest { manifest, checkpoint } => {
            cmd_validate_ingest(manifest, checkpoint)
        }
        Commands::ConvertGrok1 {
            checkpoint, manifest, output_root, format, protect_routers, protect_norms, dry_run,
        } => cmd_convert_grok1(
            checkpoint, manifest, output_root, format, protect_routers, protect_norms, dry_run,
        ),
        Commands::SmokeGrok1 {
            checkpoint, manifest, block, include_embedding, include_final_norm, output_root, dry_run,
        } => cmd_smoke_grok1(
            checkpoint, manifest, block, include_embedding, include_final_norm, output_root, dry_run,
        ),
        Commands::ValidateGrok1Artifact {
            manifest, artifact_index, checksums, output_root, strict_router_protection,
        } => cmd_validate_grok1_artifact(
            manifest, artifact_index, checksums, output_root, strict_router_protection,
        ),
        Commands::QuantizeGoz1 {
            input_dir, output, manifest, input_format, gif_threshold, use_embedded_baseline, verify,
        } => cmd_quantize_goz1(
            input_dir, output, manifest, input_format, gif_threshold, use_embedded_baseline, verify,
        ),
        Commands::Artifacts { cmd } => cmd_artifacts(cmd),
    }
}

fn cmd_validate_ingest(manifest: PathBuf, checkpoint: Option<PathBuf>) -> anyhow::Result<()> {
    artifact::validate_ingest_path(&manifest, checkpoint.as_deref())
        .map_err(|e| anyhow::anyhow!("Ingest validation failed: {}", e))?;
    println!("Ingest validation passed for {}", manifest.display());
    Ok(())
}

fn cmd_convert_grok1(
    checkpoint: Option<PathBuf>,
    manifest: PathBuf,
    output_root: PathBuf,
    format: String,
    protect_routers: bool,
    protect_norms: bool,
    dry_run: bool,
) -> anyhow::Result<()> {
    let index = artifact::convert_grok1(ConvertOptions {
        checkpoint: checkpoint.as_deref(),
        manifest: &manifest,
        output_root: &output_root,
        format: &format,
        protect_routers,
        protect_norms,
        dry_run,
    })
    .map_err(|e| anyhow::anyhow!("Grok-1 conversion failed: {}", e))?;
    println!(
        "Grok-1 conversion metadata written to {} ({} tensors, {} routers).",
        output_root.display(),
        index.tensor_count,
        index.router_count
    );
    Ok(())
}

fn cmd_smoke_grok1(
    checkpoint: Option<PathBuf>,
    manifest: PathBuf,
    block: usize,
    include_embedding: bool,
    include_final_norm: bool,
    output_root: PathBuf,
    dry_run: bool,
) -> anyhow::Result<()> {
    let index = artifact::smoke_grok1(SmokeOptions {
        checkpoint: checkpoint.as_deref(),
        manifest: &manifest,
        block,
        include_embedding,
        include_final_norm,
        output_root: &output_root,
        dry_run,
    })
    .map_err(|e| anyhow::anyhow!("Grok-1 smoke validation failed: {}", e))?;
    println!(
        "Grok-1 smoke metadata written to {} ({} tensors, {} routers).",
        output_root.display(),
        index.tensor_count,
        index.router_count
    );
    Ok(())
}

fn cmd_validate_grok1_artifact(
    manifest: PathBuf,
    artifact_index: PathBuf,
    checksums: Option<PathBuf>,
    output_root: Option<PathBuf>,
    strict_router_protection: bool,
) -> anyhow::Result<()> {
    let report = artifact::validate_grok1_artifact(
        &manifest,
        &artifact_index,
        checksums.as_deref(),
        output_root.as_deref(),
        strict_router_protection,
    )
    .map_err(|e| anyhow::anyhow!("Grok-1 artifact validation failed: {}", e))?;
    println!(
        "Grok-1 artifact validation {} ({} tensors, {} routers).",
        report.status, report.artifact_tensor_count, report.router_count
    );
    Ok(())
}

fn cmd_quantize_goz1(
    input_dir: PathBuf,
    output: PathBuf,
    manifest: Option<PathBuf>,
    input_format: CliInputFormat,
    gif_threshold: Option<f32>,
    use_embedded_baseline: bool,
    verify: bool,
) -> anyhow::Result<()> {
    let config = prepare_quantize_goz1(
        &input_dir,
        &output,
        manifest,
        input_format,
        gif_threshold,
        use_embedded_baseline,
    )?;
    let stats =
        run_quantization(&config).map_err(|e| anyhow::anyhow!("GOZ1 quantization failed: {e}"))?;
    print_quantize_goz1_summary(&output, &stats);
    maybe_verify_goz1(&output, verify)
}

/// Validate CLI paths/flags and build a [`QuantizationConfig`] for quantize-goz1.
fn prepare_quantize_goz1(
    input_dir: &Path,
    output: &Path,
    manifest: Option<PathBuf>,
    input_format: CliInputFormat,
    gif_threshold: Option<f32>,
    use_embedded_baseline: bool,
) -> anyhow::Result<QuantizationConfig> {
    if !input_dir.is_dir() {
        anyhow::bail!("--input-dir is not a directory: {}", input_dir.display());
    }
    let input_dir_s = path_to_utf8_string(input_dir, "--input-dir")?;
    let output_s = path_to_utf8_string(output, "--output")?;
    if let Some(t) = gif_threshold {
        validate_gif_threshold(t).map_err(anyhow::Error::msg)?;
    }
    // Prevent File::create(output) from truncating an input shard or the
    // classification manifest (Codex P1 on #43).
    reject_output_path_collisions(input_dir, output, manifest.as_deref())?;
    note_manifest_policy(manifest.is_none() && !use_embedded_baseline);
    Ok(quantize_goz1_config(
        input_dir_s,
        output_s,
        input_format.into(),
        manifest,
        gif_threshold,
        use_embedded_baseline,
    ))
}

fn print_quantize_goz1_summary(output: &Path, stats: &[ShardStats]) {
    let ternary: usize = stats.iter().map(|s| s.tensors_ternary).sum();
    let fp16: usize = stats.iter().map(|s| s.tensors_fp16).sum();
    let tensors = ternary + fp16;
    println!(
        "GOZ1 written to {} ({} source file(s), {} tensors: {} ternary, {} fp16/preserve; unsupported dtypes omitted).",
        output.display(),
        stats.len(),
        tensors,
        ternary,
        fp16
    );
}

fn maybe_verify_goz1(output: &Path, verify: bool) -> anyhow::Result<()> {
    if !verify {
        return Ok(());
    }
    let report =
        verify_pack_file(output).map_err(|e| anyhow::anyhow!("GOZ1 verify failed: {e}"))?;
    println!(
        "GOZ1 verify ok: version={}, {} tensor header(s), file_size={}.",
        report.version, report.tensor_count, report.file_size
    );
    Ok(())
}

fn path_to_utf8_string(path: &Path, flag: &str) -> anyhow::Result<String> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("{flag} is not valid UTF-8"))
        .map(str::to_string)
}

/// Match `resolve_manifest()`: only nonempty UTF-8 env paths count.
fn note_manifest_policy(no_explicit_manifest: bool) {
    if !no_explicit_manifest {
        return;
    }
    let env_manifest_active = std::env::var("GROK_OZEMPIC_MANIFEST")
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    if env_manifest_active {
        eprintln!("note: using GROK_OZEMPIC_MANIFEST for classification (no --manifest flag)");
    } else {
        eprintln!(
            "warning: no --manifest, no --use-embedded-baseline, and GROK_OZEMPIC_MANIFEST unset or empty; using legacy router_patterns heuristic only"
        );
    }
}

fn cmd_artifacts(cmd: ArtifactsCommands) -> anyhow::Result<()> {
    match cmd {
        ArtifactsCommands::Generate {
            manifest,
            output_dir,
            weights_dir,
            checkpoint,
        } => cmd_artifacts_generate(manifest, output_dir, weights_dir, checkpoint),
        ArtifactsCommands::Validate {
            report_dir,
            manifest,
            weights_dir,
            checkpoint,
        } => cmd_artifacts_validate(report_dir, manifest, weights_dir, checkpoint),
    }
}

fn cmd_artifacts_generate(
    manifest: PathBuf,
    output_dir: PathBuf,
    weights_dir: Option<PathBuf>,
    checkpoint: Option<String>,
) -> anyhow::Result<()> {
    println!(
        "Generating artifacts to {} using manifest {}",
        output_dir.display(),
        manifest.display()
    );
    let (actual_checkpoint, actual_shards) =
        resolve_checkpoint_and_shards(weights_dir.as_deref(), checkpoint, true)?;
    let ir = load_manifest_ir(&manifest, actual_checkpoint.as_deref(), actual_shards)?;
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
) -> anyhow::Result<()> {
    println!(
        "Validating reports in {} using manifest {}",
        report_dir.display(),
        manifest.display()
    );
    let (actual_checkpoint, actual_shards) =
        resolve_checkpoint_and_shards(weights_dir.as_deref(), checkpoint, false)?;
    let ir = load_manifest_ir(&manifest, actual_checkpoint.as_deref(), actual_shards)?;
    reports::writer::validate_report_dir_against_ir(&report_dir, &ir)
        .map_err(|e| anyhow::anyhow!("Artifact report validation failed: {}", e))?;
    println!("Report directory matches manifest and passes IR validation.");
    Ok(())
}

/// Absolute path key for collision checks (canonicalize when possible).
fn path_key(path: &Path) -> PathBuf {
    if let Ok(c) = path.canonicalize() {
        return c;
    }
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn is_quantize_weight_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    name.ends_with(".npy") || name.ends_with(".safetensors")
}

fn reject_manifest_collision(output_key: &Path, manifest: &Path) -> anyhow::Result<()> {
    if path_key(manifest) == output_key {
        anyhow::bail!(
            "--output collides with --manifest ({}); refuse to overwrite classification input",
            manifest.display()
        );
    }
    Ok(())
}

fn reject_input_weight_collisions(input_dir: &Path, output_key: &Path) -> anyhow::Result<()> {
    let rd = std::fs::read_dir(input_dir)
        .map_err(|e| anyhow::anyhow!("failed to read --input-dir {}: {e}", input_dir.display()))?;
    for entry in rd {
        let entry = entry.map_err(|e| {
            anyhow::anyhow!("failed to read --input-dir {}: {e}", input_dir.display())
        })?;
        let path = entry.path();
        if path.is_file() && is_quantize_weight_file(&path) && path_key(&path) == output_key {
            anyhow::bail!(
                "--output collides with input weight file {}; refuse to truncate quantization input",
                path.display()
            );
        }
    }
    Ok(())
}

/// Reject `--output` when it would overwrite an input weight file or the manifest.
fn reject_output_path_collisions(
    input_dir: &Path,
    output: &Path,
    manifest: Option<&Path>,
) -> anyhow::Result<()> {
    let out_key = path_key(output);
    if let Some(m) = manifest {
        reject_manifest_collision(&out_key, m)?;
    }
    reject_input_weight_collisions(input_dir, &out_key)
}

fn load_manifest_ir(
    manifest: &Path,
    actual_checkpoint: Option<&str>,
    actual_shards: Option<usize>,
) -> anyhow::Result<ArtifactIR> {
    let manifest_bytes = std::fs::read(manifest)?;
    let dissect_manifest = grok_ozempic::parse_manifest_bytes(
        &manifest_bytes,
        manifest.to_str().unwrap_or("manifest"),
    )
    .map_err(|e| anyhow::anyhow!("Failed to parse manifest: {}", e))?;

    reports::detector::build_ir_from_manifest(&dissect_manifest, actual_checkpoint, actual_shards)
        .map_err(|e| anyhow::anyhow!("Failed to build IR: {}", e))
}

/// Returns `(checkpoint_override, shard_count)` for [`reports::detector::build_ir_from_manifest`].
fn resolve_checkpoint_and_shards(
    weights_dir: Option<&Path>,
    mut checkpoint: Option<String>,
    log_shard_discovery: bool,
) -> anyhow::Result<(Option<String>, Option<usize>)> {
    let mut actual_shards = None;

    if let Some(wd) = weights_dir {
        if wd.is_dir() {
            if checkpoint.is_none() {
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
                    [] => {}
                    [only] => {
                        checkpoint = Some(only.to_string_lossy().to_string());
                    }
                    [parent, leaf] => {
                        checkpoint = Some(format!(
                            "{}/{}",
                            parent.to_string_lossy(),
                            leaf.to_string_lossy()
                        ));
                    }
                    _ => {}
                }
            }

            if let Some(count) = count_xai_tensor_shards(wd)? {
                actual_shards = Some(count);
                if log_shard_discovery {
                    println!("Discovered {} xai-dissect tensor shards.", count);
                }
            }
        } else {
            println!("Warning: weights_dir is not a valid directory.");
        }
    }

    Ok((checkpoint, actual_shards))
}

fn count_xai_tensor_shards(dir: &Path) -> anyhow::Result<Option<usize>> {
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
    Ok((count > 0).then_some(count))
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
