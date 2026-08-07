//! `quantize-goz1` command: GOZ1 packing via `run_quantization`.

use grok_ozempic::types::{
    QuantizationConfig, QuantizationInputFormat, quantize_goz1_config, validate_gif_threshold,
};
use grok_ozempic::{PackVerifyReport, ShardStats, run_quantization, verify_pack_file};
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

fn require_input_dir(input_dir: &Path) -> anyhow::Result<()> {
    if input_dir.is_dir() {
        return Ok(());
    }
    anyhow::bail!("--input-dir is not a directory: {}", input_dir.display())
}

fn optional_gif_ok(gif_threshold: Option<f32>) -> anyhow::Result<()> {
    match gif_threshold {
        Some(t) => validate_gif_threshold(t).map_err(anyhow::Error::msg),
        None => Ok(()),
    }
}

fn goz1_config_from_cli(
    input_dir_s: String,
    output_s: String,
    input_format: CliInputFormat,
    manifest: Option<PathBuf>,
    gif_threshold: Option<f32>,
    use_embedded_baseline: bool,
) -> QuantizationConfig {
    quantize_goz1_config(
        input_dir_s,
        output_s,
        QuantizationInputFormat::from(input_format),
        manifest,
        gif_threshold,
        use_embedded_baseline,
    )
}

fn utf8_io_paths(input_dir: &Path, output: &Path) -> anyhow::Result<(String, String)> {
    Ok((
        path_to_utf8_string(input_dir, "--input-dir")?,
        path_to_utf8_string(output, "--output")?,
    ))
}

/// Path UTF-8 checks, GIF flag, and output collision preflight (hard links + env manifest).
fn validate_goz1_cli_paths(
    input_dir: &Path,
    output: &Path,
    manifest: Option<&Path>,
    gif_threshold: Option<f32>,
) -> anyhow::Result<(String, String)> {
    require_input_dir(input_dir)?;
    optional_gif_ok(gif_threshold)?;
    reject_output_path_collisions(input_dir, output, manifest)?;
    utf8_io_paths(input_dir, output)
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
    let (input_dir_s, output_s) =
        validate_goz1_cli_paths(input_dir, output, manifest.as_deref(), gif_threshold)?;
    // Match resolve_manifest precedence messaging (env wins over embedded).
    note_manifest_policy(manifest.is_none(), use_embedded_baseline);
    Ok(goz1_config_from_cli(
        input_dir_s,
        output_s,
        input_format,
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
    print_applied_thresholds(&report);
    Ok(())
}

/// Surface the τ values the pack actually applied (GH #66).
///
/// The pack-level `oz.gif_threshold` metadata key records only
/// `defaults || config`, so on any run with per-tensor overrides it names a
/// threshold that some — possibly every — tensor never saw. Printing the
/// distinct applied values makes that visible at the point a human reads the
/// pack, instead of leaving `oz.gif_threshold` to be quoted as if authoritative.
fn print_applied_thresholds(report: &PackVerifyReport) {
    let Some(gifs) = report.gif_thresholds.as_deref() else {
        println!(
            "  applied gif_threshold: not recorded (container v{} predates per-tensor \
             thresholds; oz.gif_threshold metadata is NOT authoritative under per-tensor τ)",
            report.version
        );
        return;
    };
    // Ternary tensors only: fp16 rows carry 0.0 because no GIF gate ran, and
    // including them would invent a "τ = 0" tier that does not exist.
    let mut applied: Vec<f32> = gifs
        .iter()
        .copied()
        .filter(|t| *t != 0.0)
        .collect::<Vec<_>>();
    applied.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    applied.dedup();
    match applied.len() {
        0 => println!("  applied gif_threshold: none (no ternary tensors in this pack)"),
        1 => println!("  applied gif_threshold: {} (uniform)", applied[0]),
        n => {
            let list: Vec<String> = applied.iter().map(|t| t.to_string()).collect();
            println!(
                "  applied gif_threshold: {n} distinct values across ternary tensors: {}",
                list.join(", ")
            );
            println!(
                "  note: oz.gif_threshold metadata records only defaults||config and does \
                 NOT describe these -- read the tensor rows"
            );
        }
    }
}

fn path_to_utf8_string(path: &Path, flag: &str) -> anyhow::Result<String> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("{flag} is not valid UTF-8"))
        .map(str::to_string)
}

/// Match `resolve_manifest()`: only nonempty UTF-8 env paths count.
///
/// Precedence when `--manifest` is omitted: env → embedded (if requested) → legacy heuristic.
fn note_manifest_policy(no_explicit_manifest: bool, use_embedded_baseline: bool) {
    if !no_explicit_manifest {
        return;
    }
    let env_manifest_active = std::env::var("GROK_OZEMPIC_MANIFEST")
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    if env_manifest_active {
        if use_embedded_baseline {
            eprintln!(
                "note: using GROK_OZEMPIC_MANIFEST for classification (takes precedence over --use-embedded-baseline)"
            );
        } else {
            eprintln!("note: using GROK_OZEMPIC_MANIFEST for classification (no --manifest flag)");
        }
    } else if use_embedded_baseline {
        eprintln!("note: using embedded Grok-1 baseline for classification");
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

/// True when `a` and `b` name the same file identity (hard links included).
///
/// Pathname equality alone misses hard-linked aliases on Unix: two distinct
/// paths can share an inode, and `File::create` would truncate the shared data.
fn same_file_identity(a: &Path, b: &Path) -> bool {
    if path_key(a) == path_key(b) {
        return true;
    }
    same_inode_if_both_exist(a, b)
}

#[cfg(unix)]
fn same_inode_if_both_exist(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(ma), Ok(mb)) => ma.dev() == mb.dev() && ma.ino() == mb.ino(),
        _ => false,
    }
}

#[cfg(windows)]
fn same_inode_if_both_exist(a: &Path, b: &Path) -> bool {
    use std::os::windows::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(ma), Ok(mb)) => {
            ma.volume_serial_number() == mb.volume_serial_number()
                && ma.file_index() == mb.file_index()
        }
        _ => false,
    }
}

#[cfg(not(any(unix, windows)))]
fn same_inode_if_both_exist(_a: &Path, _b: &Path) -> bool {
    false
}

fn is_quantize_weight_file(path: &Path) -> bool {
    // Use OsStr extension (not full file_name UTF-8) so non-UTF-8 basenames with
    // ASCII ".npy"/".safetensors" still participate in collision checks — matching
    // stream::collect_*_shards which filter via Path::extension.
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => {
            let ext = ext.to_ascii_lowercase();
            ext == "npy" || ext == "safetensors"
        }
        None => false,
    }
}

/// Active classification manifest path (CLI first, else `GROK_OZEMPIC_MANIFEST`).
fn classification_manifest_path(cli: Option<&Path>) -> Option<PathBuf> {
    if let Some(m) = cli {
        return Some(m.to_path_buf());
    }
    match std::env::var("GROK_OZEMPIC_MANIFEST") {
        Ok(s) if !s.is_empty() => Some(PathBuf::from(s)),
        _ => None,
    }
}

fn reject_manifest_collision(output: &Path, manifest: &Path) -> anyhow::Result<()> {
    if same_file_identity(manifest, output) {
        anyhow::bail!(
            "--output collides with classification manifest ({}); refuse to overwrite",
            manifest.display()
        );
    }
    Ok(())
}

fn weight_collides_with_output(path: &Path, output: &Path) -> bool {
    path.is_file() && is_quantize_weight_file(path) && same_file_identity(path, output)
}

fn reject_input_weight_collisions(input_dir: &Path, output: &Path) -> anyhow::Result<()> {
    let rd = std::fs::read_dir(input_dir)
        .map_err(|e| anyhow::anyhow!("failed to read --input-dir {}: {e}", input_dir.display()))?;
    for entry in rd {
        let entry = entry.map_err(|e| {
            anyhow::anyhow!("failed to read --input-dir {}: {e}", input_dir.display())
        })?;
        let path = entry.path();
        if weight_collides_with_output(&path, output) {
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
    if let Some(m) = classification_manifest_path(manifest) {
        reject_manifest_collision(output, &m)?;
    }
    reject_input_weight_collisions(input_dir, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn path_key_matches_on_canonical_paths() {
        let dir = tempfile_dir();
        let a = dir.join("w.npy");
        fs::write(&a, b"x").unwrap();
        assert!(same_file_identity(&a, &a));
    }

    #[cfg(unix)]
    #[test]
    fn hard_link_alias_detected() {
        let dir = tempfile_dir();
        let weight = dir.join("w.npy");
        let alias = dir.join("alias_out.goz1");
        fs::write(&weight, b"payload").unwrap();
        std::fs::hard_link(&weight, &alias).unwrap();
        assert!(same_file_identity(&weight, &alias));
        let err = reject_input_weight_collisions(&dir, &alias).unwrap_err();
        assert!(
            err.to_string().contains("collides"),
            "expected collision, got {err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_weight_basename_still_checked() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let dir = tempfile_dir();
        let mut name = b"\xff\xfe_shard".to_vec();
        name.extend_from_slice(b".safetensors");
        let weight = dir.join(OsString::from_vec(name));
        fs::write(&weight, b"payload").unwrap();
        assert!(
            is_quantize_weight_file(&weight),
            "ASCII extension must match even when basename is non-UTF-8"
        );
        let alias = dir.join("out.goz1");
        std::fs::hard_link(&weight, &alias).unwrap();
        let err = reject_input_weight_collisions(&dir, &alias).unwrap_err();
        assert!(
            err.to_string().contains("collides"),
            "expected collision for non-UTF-8 weight hard-link, got {err}"
        );
    }

    #[test]
    fn cli_manifest_collision_detected() {
        let dir = tempfile_dir();
        let manifest = dir.join("m.json");
        fs::write(&manifest, b"{}").unwrap();
        let abs = manifest.canonicalize().unwrap();
        let err = reject_output_path_collisions(&dir, &abs, Some(&abs)).unwrap_err();
        assert!(
            err.to_string().contains("collides"),
            "expected cli manifest collision, got {err}"
        );
    }

    #[test]
    fn classification_manifest_path_prefers_cli() {
        let cli = PathBuf::from("cli-manifest.json");
        assert_eq!(classification_manifest_path(Some(&cli)), Some(cli));
    }

    /// Env-var collision uses the same `reject_manifest_collision` path as CLI
    /// once `classification_manifest_path` returns a path. Avoid `set_var` in
    /// tests (Semgrep unsafe-usage); cover identity via explicit CLI path.
    #[test]
    fn resolved_manifest_path_collides_with_output() {
        let dir = tempfile_dir();
        let manifest = dir.join("env-or-cli.json");
        fs::write(&manifest, b"{}").unwrap();
        let abs = manifest.canonicalize().unwrap();
        // Same check path as when classification_manifest_path(None) returns env.
        let err = reject_manifest_collision(&abs, &abs).unwrap_err();
        assert!(
            err.to_string().contains("collides"),
            "expected manifest/output identity collision, got {err}"
        );
    }

    fn tempfile_dir() -> PathBuf {
        // Prefer crate target/ over shared OS temp_dir (Semgrep temp-dir).
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("quantize-collision-tests")
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
}
