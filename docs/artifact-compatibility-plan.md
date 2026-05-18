# grok-ozempic ↔ xai-dissect Artifact Compatibility Plan

## 1. Schema Contract & JSON/Intermediate Representation

To ensure `grok-ozempic` produces artifacts structurally identical to `xai-dissect`, we define a strict internal intermediate representation (IR) that serializes natively to the expected markdown. 

```rust
// src/reports/schema.rs
pub struct ArtifactManifest {
    pub model_family: String, // "grok-1"
    pub checkpoint: String,
    pub shards: usize,        // 770
    pub schema_version: u32,  // 1
}

pub struct Hyperparameters {
    pub vocab_size: usize, // 131072
    pub d_model: usize,    // 6144
    pub n_experts: usize,  // 8
    pub d_ff: usize,       // 32768
    pub n_blocks: usize,   // 64
}

pub struct TensorTotals {
    pub total: usize,
    pub f32_tensors: usize,
    pub int8_tensors: usize,
    pub quant_tensors: usize,
}

pub struct RouterEntry {
    pub block: usize,
    pub slot: usize, // 11
    pub shape: (usize, usize), // (6144, 8)
    pub orientation: String, // "d_model_to_experts"
    pub experts: usize, // 8
    pub kind: String, // "router"
    pub structural_name: String, // "block_{:03}.routing_slot_11"
}

pub struct ExpertBlock {
    pub block: usize,
    pub experts: usize, // 8
    pub expert_tensors: usize, // 3
    pub slots: Vec<usize>, // [0, 1, 2]
    pub shapes: Vec<String>, // ["expert_slot_00 (8, 6144, 32768)", ...]
}

pub struct SaaqTarget {
    pub rank: usize,
    pub tensor: String,
    pub kind: String,
    pub region: String,
    pub readiness: f64,
    pub opportunity: f64,
    pub risk: f64,
    pub disposition: String,
}

pub struct SaaqCritical {
    pub tensor: String,
    pub readiness: f64,
    pub risk: f64,
    pub reasons: String, // "distribution=dense_balanced<br>..."
}

pub struct StatsEntry {
    pub tensor: String,
    pub kind: String,
    pub block: usize,
    pub value: f64, // RMS
}
```

## 2. Canonical Tensor Naming & Classification Rules

Mapping `grok-ozempic` physical tensor states to `xai-dissect` logical artifact names:

| Physical/Logical Kind | Expected Artifact Name | Properties |
| --- | --- | --- |
| `token_embedding` | `embedding.slot_00.token_embedding` | Shape `(131072, 6144)` |
| `router` | `block_{:03}.routing_slot_11` | Shape `(6144, 8)` |
| `expert_00` | `expert_slot_00` | Shape `(8, 6144, 32768)` (up / w1) |
| `expert_01` | `expert_slot_01` | Shape `(8, 32768, 6144)` (down / w2) |
| `expert_02` | `expert_slot_02` | Shape `(8, 6144, 32768)` (gate / w3) |

## 3. Routing Detection Rules & Invariants

- **Rule:** Filter tensor records matching segment globs `*.*.attn_router.weight` or legacy `router` substrings.
- **Invariants to enforce:**
  1. Exactly `n_blocks` (64) routers must exist.
  2. The shape must strictly be `(6144, 8)`.
  3. The orientation must be labeled `d_model_to_experts`.
  4. The artifact slot must be `11`.
  5. The structural name format must strictly be `block_{:03}.routing_slot_11`.

## 4. Expert Detection Rules & Invariants

- **Rule:** Filter tensor records matching expert projection shapes within a blocked loop.
- **Invariants to enforce:**
  1. Each block must have exactly 3 expert tensors.
  2. `expert_slot_00` must map to `(8, d_model, d_ff)` -> `(8, 6144, 32768)`.
  3. `expert_slot_01` must map to `(8, d_ff, d_model)` -> `(8, 32768, 6144)`.
  4. `expert_slot_02` must map to `(8, d_model, d_ff)` -> `(8, 6144, 32768)`.
  5. The total expert count per block must be `8`.

## 5. Markdown Output Templates

Reports must match the upstream headers precisely.

**`inventory.md`**
```markdown
# xai-dissect inventory

- **model_family**: `{model_family}`
- **checkpoint**: `{checkpoint}`
- **shards**: {shards}
- **schema_version**: 1

## Inferred hyperparameters

| Field | Value |
| ----- | ----- |
| vocab_size | {vocab_size} |
| d_model | {d_model} |
...
```

**`routing-report.md`**
```markdown
# xai-dissect routing report

- **model_family**: `{model_family}`
- **checkpoint**: `{checkpoint}`
- **shards**: {shards}
- **relevant_blocks**: 64
- **expected_experts_per_router**: 8
- **schema_version**: 1

## Candidate routing tensors

| Block | Slot | Shape | Orientation | Experts | Kind | Structural name |
| ----: | ---: | ----- | ----------- | ------: | ---- | --------------- |
| {block} | {slot} | `{shape}` | {orientation} | {experts} | {kind} | `{structural_name}` |
```

**`experts.md`**
```markdown
# xai-dissect expert atlas

...

## Expert counts by block

| Block | Experts | Expert tensors | Slots | Shapes |
| ----: | ------: | -------------: | ----- | ------ |
| {block} | {experts} | {expert_tensors} | {slots} | {shapes_html_br} |
```

## 6. Validator Tests Requirements

- `test_inventory_totals`: Verify tensors parse and total exactly 770.
- `test_router_shape_strict`: Inject an invalid `(8, 6144)` router shape into the IR and verify the validator strictly rejects it.
- `test_expert_slot_mapping`: Verify that the order `00, 01, 02` strictly aligns with the shapes `(8, 6144, 32768)`, `(8, 32768, 6144)`, and `(8, 6144, 32768)`.
- `test_saaq_readiness_criticality`: Ensure `token_embedding` is listed as a candidate and all 64 routers are listed as high-risk/critical.

## 7. Failure Modes

1. **Incomplete Checkpoint Stream:** Emitting a report where the sum of tensors is less than 770.
2. **Dimension Transposition:** Accidental transposition of `.npy` shapes mapping the router as `(8, 6144)`. The validator *must* halt report generation.
3. **Missing Legacy Slot Fallbacks:** The legacy xai-dissect format used hardcoded slot index strings (`slot_11`, `slot_00`, `slot_01`, `slot_02`). Missing these exactly will break downstream regex parsers.
4. **HTML Markdown Escaping:** The `experts.md` table uses `<br>` tags to separate the shapes in a single cell. Standard markdown renderers might escape this if generating via a strict markdown AST builder. Must use raw string interpolation to ensure the literal `<br>` is present.

## 8. CLI Shape Integration

To stay decoupled from the core quantization loop, we expose this via a new subcommand:

```bash
grok-ozempic artifacts generate \
    --manifest path/to/dissect.json \
    --output-dir ./reports/grok-1-official__ckpt-0

grok-ozempic artifacts validate \
    --report-dir ./reports/grok-1-official__ckpt-0
```

## 9. Exact Cursor Patch Checklist

- [ ] 1. Add `clap` CLI entrypoint in `src/bin/grok-ozempic.rs` (or equivalent binary target) exposing the `artifacts` subcommand.
- [ ] 2. Create `src/reports/mod.rs` to expose the new reporting module.
- [ ] 3. Create `src/reports/schema.rs` with the `ArtifactManifest`, `Hyperparameters`, `RouterEntry`, and other structs defined in Section 1.
- [ ] 4. Create `src/reports/detector.rs` that reads a `GOZ1` stream or manifest, applies the classification rules (Section 2/3/4), and constructs the IR structs.
- [ ] 5. Create `src/reports/templates.rs` providing `format!()` macros for `inventory.md`, `routing-report.md`, `experts.md`, `saaq-readiness.md`, and `stats.md` matching Section 5 exactly.
- [ ] 6. Create `src/reports/writer.rs` that writes the interpolated templates to the `--output-dir`.
- [ ] 7. Create `src/reports/validator.rs` implementing strict invariant checks (770 tensors, shapes, block counts) that either panic or return `Err` if the generated IR violates the schema.
- [ ] 8. Add unit tests in `src/reports/tests.rs` covering the requirements in Section 6.

## 10. Open Questions

- Should `stats.md` compute real RMS values from the `.goz1` tensor payloads or proxy dummy values for structural compatibility if payloads aren't fully read during artifact generation? For now, we will pass through `xai-dissect` values if parsing an upstream manifest, or compute naive estimates if processing a raw checkpoint.