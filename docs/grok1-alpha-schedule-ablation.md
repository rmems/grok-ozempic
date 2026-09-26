# Grok-1 matched alpha/schedule experiment (GH125)

This is an implementation runbook, not measured four-cell evidence. GH85's
three-arm inventory, guard rejecting its absmax/HP123 command, and canonical
Option 2 report remain unchanged. Never combine historical GH85 measurements
with new cells as a same-run comparison. The new protocol is
`grok1-int4-alpha-schedule-v1`.

| Cell | Quantized scale mode | FP16 expert blocks |
|---|---|---|
| A | absmax | none |
| B | weight-LS channel-alpha | none |
| C | absmax | 1,2,3 |
| D | weight-LS channel-alpha | 1,2,3 |

Every cell fixes 8192 sampled token IDs, seed 20260806, blocks 0–3 and top-k 2.
The ordered little-endian int64 token array has SHA256
`57b0e364bb25ac4bd9047b592e28ae1470af94481be073a1f963c7cd0a5ee3de`.
IDs and their digest are retained. Each arm carries its own residual throughout
the chain. Attention, routers, norms and the reference remain high precision.
The mandatory `fp16_roundtrip` control rounds requested weights through FP16
back to FP32; this is weight-precision sensitivity, not native FP16 compute,
an expert-only ceiling, or upstream implementation parity.

## Qualification and launch

Run the registered Python suite in the task worktree before launch:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 just _python-tests
ruff check scripts/grok1_alpha_schedule_*.py scripts/test_grok1_alpha_schedule_*.py
```

Commit the verified implementation before measurement:

```bash
git add scripts/grok1_alpha_schedule_*.py scripts/test_grok1_alpha_schedule_*.py docs/grok1-alpha-schedule-ablation.md justfile .github/workflows/python-scripts.yml CLAUDE.md
git commit -m "Add matched Grok-1 alpha and schedule ablation harness" --trailer "Co-authored-by: Codex <noreply@openai.com>"
git status --porcelain --untracked-files=all -- scripts src
```

The final command must be empty. The launcher independently verifies a full
clean implementation commit and the source bytes loaded by its processes;
there is no flag to waive provenance, controls, fixed budgets or launch gates.
Use a new output directory; canonical precision-remedy report trees are rejected.
The existing path/pattern conventions and absolute patterns are supported.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/grok1_alpha_schedule_ablation.py \
  --npy-root /home/raulmc/.models/xai-grok-1/export-npy \
  --pack-root /home/raulmc/.models/xai-grok-1/artifacts/multiblock-68 \
  --embedding-shard /home/raulmc/.models/xai-grok-1/export-npy/embedding__slot_00__token_embedding.npy \
  --out reports/grok-1-int4-alpha-schedule-ablation \
  --preflight-only
```

Preflight reads array headers and checks resources and implementation; it does
not perform a forward or claim whole-file input verification. Only after it
passes, launch with the identical command **without `--preflight-only`**.
Destination/source overlap is rejected before locks, directory creation or
publication, including preflight-only mode. Published preflight records redact
known host paths; disk requirements retain device identifiers and byte reservations,
not the local paths used for the free-space probes. Launch manifests remain local
execution records with resolved paths, not portable public preflight reports.
An optional `--int4-side-root /dedicated/cache/path` reuses content-bound caches.
Default caches live below the new output directory. The internal `--cell`
entry point requires a launch manifest and live parent supervisor.

Before each child, including the child-side check, require MemAvailable at
least 24 GiB, free swap at least 2 GiB, and 40 GiB free disk **plus** calculated
additional output space on each writable filesystem. Shared filesystems sum
the reservations. Cache reserves cover complete replacement generations of
int8 codes plus both scale tables, headers and binding metadata; report scratch
adds 64 MiB. The process census blocks other processes with RSS at least 2 GiB.
Also ensure no competing heavyweight CPU/GPU job: the RSS census cannot detect
every low-RSS GPU workload. No 2048-token fallback is available.

## Evidence and failure semantics

Each invocation creates fresh `runs/<run-id>/{A,B,C,D}/` paths with launch
manifests, progress, child logs, execution observations, raw metrics and
validated metrics. The authoritative `outcome.json` is atomically marked
inconclusive before expensive work begins. Only all four valid fresh cells
produce a complete outcome and paired contrasts. `results.md` renders that
record; partial output is retained on failure, and old run directories remain
auditable. Restart starts a new run and reuses only verified weight caches.
Output locks prevent concurrent supervisors from owning one report directory.
Catchable interruptions during handler installation, directory creation or the
first authoritative write also publish an inconclusive result under that lock.
No cleanup can guarantee replacement during uncatchable SIGKILL, host loss or
an unwritable filesystem before the first authoritative write.

Children have a fixed 24-hour lifetime. Catchable supervisor interrupts kill
and reap the child process group; nonzero exits, kill signals, missing or
malformed output, changed inputs, missing controls or terminal drift all yield
nonzero/inconclusive results. Uncatchable supervisor SIGKILL or host loss cannot
run Python cleanup; the already-published inconclusive state prevents a false
success. Inspect and stop any surviving child before retrying after SIGKILL.

All NPY block content, pack bytes and embedding bytes are bound to the run.
The checkpoint identity is a digest of these measured inputs, not an attestation
of an upstream checkpoint version. Block sources are checked around their
forwards, embedding bytes around each chain, and implementation/runtime must
match across all cells. Side tables retain the existing approximately 64 MiB
whole-row construction and content-bound generation semantics.

Reports retain per-block top-1/top-2 agreement, margin-stratified flips,
MoE/output cosine, incoming drift, post-block-3 drift and FP16 controls.
Every metric gets B−A, D−C, C−A, D−B and (D−C)−(B−A), favorable directions and
observed effect descriptions. Empty margin bands have zero counts and null
rates with an explicit unavailable reason, rather than fabricated rates.
The 0.95 top-1 band is diagnostic, not deployment/language-quality certification.
Signs describe one run, not statistical significance; a deterministic repeat
tests reproducibility, not independent statistical evidence.

Resource fields distinguish logical four-bit codes, actual int8 NPY payload
and file bytes, float32 scales, FP16 expert payload, nonexpert high-precision
payload, measured-block total, cache/scratch file bytes, elapsed child wall
time and sampled process-group aggregate RSS. RSS is sampled every 50 ms and
can miss shorter peaks; shared pages may be counted more than once. Runtime
FP32 dequantization/reference allocations and cache retention are not the
logical precision payload. Disk totals sum file `stat().st_size` at child
completion before final metric publication; they exclude filesystem allocation
overhead and later report writes. In the normal launcher, cache is
`out/int4-side` and each child's scratch is `out/runs/<run-id>/<cell>`: these
are disjoint trees, so cache bytes are not included in the scratch total.
`actual_resources()` reports each supplied tree independently; direct callers
passing overlapping trees must not add those disk totals as distinct storage.
Missing positive process-group RSS remains invalid evidence; direct-child RSS
is not silently substituted. A secondary child-cleanup timeout is reported to
stderr without masking the original failure (and is fatal when no earlier failure
exists). Hypothetical nibble packing is explicitly separate
and is not a GOZ1 format or implemented side-table representation. A mixed
schedule's quality gain is not automatically an equal-byte compression win.

Completing this harness does not close GH125's measured-experiment acceptance
items. A clean, gated four-cell launch and independent artifact review remain
necessary.
