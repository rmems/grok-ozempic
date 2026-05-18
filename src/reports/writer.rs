use crate::reports::schema::ArtifactIR;
use crate::reports::templates;
use std::fs;
use std::path::Path;

pub fn write_reports(ir: &ArtifactIR, output_dir: &Path) -> std::io::Result<()> {
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    fs::write(
        output_dir.join("inventory.md"),
        templates::generate_inventory(ir),
    )?;

    fs::write(
        output_dir.join("routing-report.md"),
        templates::generate_routing_report(ir),
    )?;

    fs::write(
        output_dir.join("experts.md"),
        templates::generate_experts_report(ir),
    )?;

    fs::write(
        output_dir.join("saaq-readiness.md"),
        templates::generate_saaq_readiness(ir),
    )?;

    fs::write(output_dir.join("stats.md"), templates::generate_stats(ir))?;

    Ok(())
}
