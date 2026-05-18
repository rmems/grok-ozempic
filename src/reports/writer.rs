use crate::error::GrokOzempicError;
use crate::reports::schema::ArtifactIR;
use crate::reports::templates;
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

pub fn write_reports(ir: &ArtifactIR, output_dir: &Path) -> std::io::Result<()> {
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
