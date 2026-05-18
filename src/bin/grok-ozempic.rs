use clap::{Parser, Subcommand};
use grok_ozempic::reports;
use std::path::PathBuf;

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
            } => {
                println!(
                    "Generating artifacts to {} using manifest {}",
                    output_dir.display(),
                    manifest.display()
                );

                // Let's implement this for real
                let manifest_bytes = std::fs::read(&manifest)?;
                let dissect_manifest = grok_ozempic::parse_manifest_bytes(
                    &manifest_bytes,
                    manifest.to_str().unwrap_or("manifest"),
                )
                .map_err(|e| anyhow::anyhow!("Failed to parse manifest: {}", e))?;

                let ir = reports::detector::build_ir_from_manifest(&dissect_manifest)
                    .map_err(|e| anyhow::anyhow!("Failed to build IR: {}", e))?;

                reports::validator::validate_ir(&ir)
                    .map_err(|e| anyhow::anyhow!("Artifact validation failed: {}", e))?;

                reports::writer::write_reports(&ir, &output_dir)
                    .map_err(|e| anyhow::anyhow!("Failed to write reports: {}", e))?;

                println!("Success!");
            }
            ArtifactsCommands::Validate { report_dir } => {
                println!("Validating reports in {}", report_dir.display());
                // In a complete implementation, this would read back and validate:
            }
        },
    }

    Ok(())
}
