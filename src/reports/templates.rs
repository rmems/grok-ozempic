use crate::reports::schema::ArtifactIR;
use std::fmt::Write as _;

const DERIVED_REPORT_SCHEMA_VERSION: u32 = 1;

pub fn generate_inventory(ir: &ArtifactIR) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        r#"# xai-dissect inventory

- **model_family**: `{}`
- **checkpoint**: `{}`
- **shards**: {}
- **schema_version**: {}

## Inferred hyperparameters

| Field | Value |
| ----- | ----- |
| vocab_size | {} |
| d_model | {} |
| n_experts | {} |
| d_ff | {} |
| n_blocks | {} |

## Totals

| Metric | Value |
| ------ | ----- |
| tensors | {} |
| f32 tensors | {} |
| int8 tensors | {} |
| quant tensors | {} |
| total elements | {} |
| total bytes | {} ({}) |
"#,
        ir.manifest.model_family,
        ir.manifest.checkpoint,
        ir.manifest.shards,
        ir.manifest.schema_version,
        ir.hyperparameters.vocab_size,
        ir.hyperparameters.d_model,
        ir.hyperparameters.n_experts,
        ir.hyperparameters.d_ff,
        ir.hyperparameters.n_blocks,
        ir.totals.total,
        ir.totals.f32_tensors,
        ir.totals.int8_tensors,
        ir.totals.quant_tensors,
        ir.totals.total_elements,
        ir.totals.total_bytes,
        human_bytes(ir.totals.total_bytes),
    );

    let _ = write!(
        out,
        r#"
## Tensor kinds

| Kind | Count | Bytes |
| ---- | ----: | ----: |
"#
    );
    for kind in &ir.inventory_kinds {
        let _ = writeln!(
            out,
            "| {} | {} | {} ({}) |",
            kind.kind,
            kind.count,
            kind.bytes,
            human_bytes(kind.bytes)
        );
    }

    let _ = write!(
        out,
        r#"
## Blocks

| Label | Block | Shards | Tensors | Bytes | Kinds |
| ----- | ----: | ------ | ------: | ----: | ----- |
"#
    );
    for block in &ir.inventory_blocks {
        let kinds = block
            .kinds
            .iter()
            .map(|kind| format!("{}x{}", kind.count, kind.kind))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "| {} | {} | {}..={} | {} | {} ({}) | {} |",
            block.label,
            block
                .block
                .map(|block| block.to_string())
                .unwrap_or_else(|| "-".to_string()),
            block.shard_start,
            block.shard_end,
            block.tensors,
            block.bytes,
            human_bytes(block.bytes),
            kinds
        );
    }

    if !ir.exemplar_tensors.is_empty() {
        let _ = write!(
            out,
            r#"
## Exemplar block (`block_000`)

| Shard | In-shard | Role | Dtype | Shape | Kind | Slot |
| ----: | -------: | ---- | ----- | ----- | ---- | ---: |
"#
        );
        for tensor in &ir.exemplar_tensors {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | `{}` | {} | {} |",
                tensor.shard,
                tensor.in_shard,
                tensor.role,
                tensor.dtype,
                tensor.shape,
                tensor.kind,
                tensor
                    .slot
                    .map(|slot| slot.to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
    }
    out
}

pub fn generate_routing_report(ir: &ArtifactIR) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        r#"# xai-dissect routing report

- **model_family**: `{}`
- **checkpoint**: `{}`
- **shards**: {}
- **relevant_blocks**: {}
- **expected_experts_per_router**: {}
- **schema_version**: {}

## Candidate routing tensors

| Block | Slot | Shape | Orientation | Experts | Kind | Structural name |
| ----: | ---: | ----- | ----------- | ------: | ---- | --------------- |
"#,
        ir.manifest.model_family,
        ir.manifest.checkpoint,
        ir.manifest.shards,
        ir.hyperparameters.n_blocks,
        ir.hyperparameters.n_experts,
        DERIVED_REPORT_SCHEMA_VERSION
    );

    for r in &ir.routers {
        let _ = writeln!(
            out,
            "| {} | {} | `({}, {})` | {} | {} | {} | `{}` |",
            r.block,
            r.slot,
            r.shape.0,
            r.shape.1,
            r.orientation,
            r.experts,
            r.kind,
            r.structural_name
        );
    }
    out
}

pub fn generate_experts_report(ir: &ArtifactIR) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        r#"# xai-dissect expert atlas

- **model_family**: `{}`
- **checkpoint**: `{}`
- **shards**: {}
- **relevant_blocks**: {}
- **expected_experts_per_block**: {}
- **schema_version**: {}

## Expert counts by block

| Block | Experts | Expert tensors | Slots | Shapes |
| ----: | ------: | -------------: | ----- | ------ |
"#,
        ir.manifest.model_family,
        ir.manifest.checkpoint,
        ir.manifest.shards,
        ir.hyperparameters.n_blocks,
        ir.hyperparameters.n_experts,
        DERIVED_REPORT_SCHEMA_VERSION
    );

    for b in &ir.expert_blocks {
        let slots_str = b
            .slots
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let shapes_str = b.shapes.join("<br>");
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            b.block, b.experts, b.expert_tensors, slots_str, shapes_str
        );
    }
    out
}

pub fn generate_saaq_readiness(ir: &ArtifactIR) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        r#"# xai-dissect SAAQ-readiness report

- **model_family**: `{}`
- **checkpoint**: `{}`
- **shards**: {}
- **candidate_targets**: {}
- **routing_critical_tensors**: {}
- **schema_version**: {}

## Candidate target tensors

| Rank | Tensor | Kind | Region | Readiness | Opportunity | Risk | Disposition |
| ---: | ------ | ---- | ------ | --------: | ----------: | ---: | ----------- |
"#,
        ir.manifest.model_family,
        ir.manifest.checkpoint,
        ir.manifest.shards,
        ir.saaq_targets.len(),
        ir.saaq_critical.len(),
        DERIVED_REPORT_SCHEMA_VERSION
    );

    for t in &ir.saaq_targets {
        let _ = writeln!(
            out,
            "| {} | `{}` | {} | {} | {:.3} | {:.3} | {:.3} | {} |",
            t.rank, t.tensor, t.kind, t.region, t.readiness, t.opportunity, t.risk, t.disposition
        );
    }

    let _ = write!(
        out,
        r#"
## Routing-critical tensors

| Tensor | Readiness | Risk | Reasons |
| ------ | --------: | ---: | ------- |
"#
    );

    for c in &ir.saaq_critical {
        let _ = writeln!(
            out,
            "| `{}` | {:.3} | {:.3} | {} |",
            c.tensor, c.readiness, c.risk, c.reasons
        );
    }
    out
}

pub fn generate_stats(ir: &ArtifactIR) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        r#"# xai-dissect stats report

- **model_family**: `{}`
- **checkpoint**: `{}`
- **shards**: {}
- **sample_values_per_tensor**: 65536
- **schema_version**: {}

## Norm summary

- **mean_rms**: {:.6}

### Top RMS tensors

| Tensor | Kind | Block | Value |
| ------ | ---- | ----: | ----: |
"#,
        ir.manifest.model_family,
        ir.manifest.checkpoint,
        ir.manifest.shards,
        DERIVED_REPORT_SCHEMA_VERSION,
        ir.mean_rms
    );

    for s in &ir.stats {
        let _ = writeln!(
            out,
            "| `{}` | {} | {} | {:.6} |",
            s.tensor, s.kind, s.block, s.value
        );
    }
    out
}

fn human_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;

    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.2} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.2} KiB", bytes_f / KIB)
    } else if bytes == 1 {
        "1 B".to_string()
    } else {
        format!("{bytes} B")
    }
}
