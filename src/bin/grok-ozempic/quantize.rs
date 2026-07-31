//! `quantize-goz1` command: GOZ1 packing via `run_quantization`.

use grok_ozempic::types::{
    QuantizationConfig, QuantizationInputFormat, quantize_goz1_config, validate_gif_threshold,
};
use grok_ozempic::{ShardStats, run_quantization, verify_pack_file};
use std::path::{Path, PathBuf};

use crate::CliInputFormat;

pub(crate) fn cmd_quantize_goz1(
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
        QuantizationInputFormat::from(input_format),
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
