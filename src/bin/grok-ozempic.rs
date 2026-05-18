use clap::{Parser, Subcommand};
use grok_ozempic::reports;
use std::path::{Path, PathBuf};

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
    /// Validate generated reports in a directory
    Validate {
        /// Directory containing generated reports to validate
        #[arg(long)]
        report_dir: PathBuf,
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

                let mut actual_checkpoint = checkpoint;
                let mut actual_shards = None;

                if let Some(wd) = weights_dir {
                    if wd.is_dir() {
                        // Derive checkpoint provenance from path
                        if actual_checkpoint.is_none() {
                            let comps: Vec<_> = wd.components().rev().take(2).collect();
                            if comps.len() == 2 {
                                actual_checkpoint = Some(format!(
                                    "{}/{}",
                                    comps[1].as_os_str().to_string_lossy(),
                                    comps[0].as_os_str().to_string_lossy()
                                ));
                            } else if comps.len() == 1 {
                                actual_checkpoint =
                                    Some(comps[0].as_os_str().to_string_lossy().to_string());
                            }
                        }

                        if let Some(count) = count_xai_tensor_shards(&wd)? {
                            actual_shards = Some(count);
                            println!("Discovered {} xai-dissect tensor shards.", count);
                        }
                    } else {
                        println!("Warning: weights_dir is not a valid directory.");
                    }
                }

                let manifest_bytes = std::fs::read(&manifest)?;
                let dissect_manifest = grok_ozempic::parse_manifest_bytes(
                    &manifest_bytes,
                    manifest.to_str().unwrap_or("manifest"),
                )
                .map_err(|e| anyhow::anyhow!("Failed to parse manifest: {}", e))?;

                let ir = reports::detector::build_ir_from_manifest(
                    &dissect_manifest,
                    actual_checkpoint.as_deref(),
                    actual_shards,
                )
                .map_err(|e| anyhow::anyhow!("Failed to build IR: {}", e))?;

                reports::validator::validate_ir(&ir)
                    .map_err(|e| anyhow::anyhow!("Artifact validation failed: {}", e))?;

                reports::writer::write_reports(&ir, &output_dir)
                    .map_err(|e| anyhow::anyhow!("Failed to write reports: {}", e))?;

                println!("Success!");
            }
            ArtifactsCommands::Validate { report_dir } => {
                println!("Validating reports in {}", report_dir.display());
                reports::writer::validate_report_dir(&report_dir)
                    .map_err(|e| anyhow::anyhow!("Artifact report validation failed: {}", e))?;
                println!("Report directory is structurally valid.");
            }
        },
    }

    Ok(())
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
