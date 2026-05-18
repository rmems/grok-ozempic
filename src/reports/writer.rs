use crate::error::GrokOzempicError;
use crate::reports::schema::ArtifactIR;
use crate::reports::templates;
use crate::reports::validator;
use std::fs;
use std::path::Path;

/// Markdown files produced by [`write_reports`], used to verify a report directory.
const REPORT_MARKDOWN_FILES: &[&str] = &[
    "inventory.md",
    "routing-report.md",
    "experts.md",
    "saaq-readiness.md",
    "stats.md",
];

/// Ensures `dir` exists, is a directory, and contains the report files from [`write_reports`].
pub fn validate_report_dir(dir: &Path) -> Result<(), GrokOzempicError> {
    if !dir.exists() {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "report directory does not exist: {}",
            dir.display()
        )));
    }
    if !dir.is_dir() {
        return Err(GrokOzempicError::ArtifactValidation(format!(
            "report path is not a directory: {}",
            dir.display()
        )));
    }
    for name in REPORT_MARKDOWN_FILES {
        let path = dir.join(name);
        if !path.is_file() {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "missing expected report file: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Like [`validate_report_dir`], then ensures markdown is non-empty, `ir` passes [`validator::validate_ir`],
/// and each report file matches the canonical template output for `ir`.
pub fn validate_report_dir_against_ir(dir: &Path, ir: &ArtifactIR) -> Result<(), GrokOzempicError> {
    validate_report_dir(dir)?;
    validator::validate_ir(ir)?;

    let expected: [(&str, String); 5] = [
        (REPORT_MARKDOWN_FILES[0], templates::generate_inventory(ir)),
        (
            REPORT_MARKDOWN_FILES[1],
            templates::generate_routing_report(ir),
        ),
        (
            REPORT_MARKDOWN_FILES[2],
            templates::generate_experts_report(ir),
        ),
        (
            REPORT_MARKDOWN_FILES[3],
            templates::generate_saaq_readiness(ir),
        ),
        (REPORT_MARKDOWN_FILES[4], templates::generate_stats(ir)),
    ];

    for (name, expected_body) in expected {
        let path = dir.join(name);
        let got = fs::read_to_string(&path)?;
        if got.trim().is_empty() {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "report file is empty: {}",
                path.display()
            )));
        }
        if got != expected_body {
            return Err(GrokOzempicError::ArtifactValidation(format!(
                "report file content does not match expected artifact IR (file: {})",
                path.display()
            )));
        }
    }

    Ok(())
}

pub fn write_reports(ir: &ArtifactIR, output_dir: &Path) -> Result<(), GrokOzempicError> {
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    fs::write(
        output_dir.join(REPORT_MARKDOWN_FILES[0]),
        templates::generate_inventory(ir),
    )?;

    fs::write(
        output_dir.join(REPORT_MARKDOWN_FILES[1]),
        templates::generate_routing_report(ir),
    )?;

    fs::write(
        output_dir.join(REPORT_MARKDOWN_FILES[2]),
        templates::generate_experts_report(ir),
    )?;

    fs::write(
        output_dir.join(REPORT_MARKDOWN_FILES[3]),
        templates::generate_saaq_readiness(ir),
    )?;

    fs::write(
        output_dir.join(REPORT_MARKDOWN_FILES[4]),
        templates::generate_stats(ir),
    )?;

    Ok(())
}
