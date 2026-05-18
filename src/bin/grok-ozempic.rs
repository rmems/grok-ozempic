use clap::{Parser, Subcommand};
use grok_ozempic::reports;
use grok_ozempic::reports::schema::ArtifactIR;
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
    let cli = Cli::parse();

    match cli.command {
        Commands::Artifacts { cmd } => match cmd {
            ArtifactsCommands::Generate {
                manifest,
                output_dir,
                weights_dir,
                checkpoint,
            } => {
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
            }
            ArtifactsCommands::Validate {
                report_dir,
                manifest,
                weights_dir,
                checkpoint,
            } => {
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
            }
        },
    }

    Ok(())
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
