# GH125 clean-implementation preflight — 2026-09-22

**Agent:** Codex (OpenAI).

Status: **measurement not launched**. This is a header/resource preflight,
not a completed experiment or an accepted scientific inconclusive outcome.
GH125 / RM-1024 measured acceptance remains open. No historical GH85 metric
was substituted for A, B, C, D or a precision control.

The launcher returned exit 1: `free swap below 2 GiB`. Its JSON outputs are
preserved alongside this note. No model forward or SAAQ threshold sweep ran.
Privacy correction, 2026-09-22: the machine-local disk probe `path` was removed
from `resource-preflight.json`, and the reproduction example now uses a supplied
model root. All observed values, device identity, implementation identity and
failure status remain unchanged. This redaction is not a new measurement.

## Implementation and qualification

- Clean implementation commit: `89bfefb1bee3c7666404db23952a44190f638220`.
- An independent `clean_implementation()` call returned that exact commit and
  `dirty: false`, checking committed/resident/on-disk Python source bytes.
- Runtime: Python 3.14.6, NumPy 2.5.1,
  `Linux-7.1.5-200.fc44.x86_64-x86_64-with-glibc2.43`.
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.
- Full `just review` passed: 495 Python tests across 18 registered modules,
  218 Rust unit tests, 17 Rust integration tests, formatting, all-feature
  clippy/build/docs, hook tests, actionlint, Ruff, shellcheck and py_compile.
- `cargo-audit` was skipped by the recipe because it is not installed.
- Earlier hook-test attempts passed assertions but failed temporary-directory
  cleanup while `.git/ai` was being populated; the unchanged full gate passed
  on retry. No check was weakened or bypassed.

## Resource observation

Observed approximately 2026-09-22 01:29 CDT; these values are not reusable
authorization for a future launch.

| Gate | Observed | Requirement | Result |
|---|---:|---:|---|
| MemAvailable | 26,677,383,168 bytes (24.85 GiB) | 24 GiB | Pass |
| Free swap | 151,552 bytes (148 KiB) | 2 GiB | **Fail** |
| Free disk | 68,888,530,944 bytes (64.16 GiB) | 40 GiB + calculated additional outputs | Pass |
| Additional output reservation | 19,412,860,928 bytes (18.08 GiB) | Combined on filesystem 49 | Included |
| RSS competitor census | PID 10387 | No process outside supervisor/parent at or above 2 GiB | **Flagged** |

The disk floor plus reservation is 62,362,533,888 bytes (58.08 GiB).
PID 10387 was identified read-only as `ChatGPT`; a subsequent `ps` sample
reported 3,607,888 KiB RSS. This is a conservative RSS gate flag, not proof
that the application is a GPU training workload. It was not terminated.

A supplemental GPU snapshot showed RTX 5080 utilization 23%, 1,675 MiB used,
46 C, fan 30%, and 71.97 W; the compute-process query listed only
`kwin_wayland` and `ptyxis`. The current one-second vmstat interval showed
91% CPU idle and no swap-in/out. These samples do not override the failed
swap gate or RSS census. No swap reset/disable, process kill, or system
configuration change was made to make the gate pass.

## Reproduction and remaining work

From a clean checkout containing the implementation, with the existing NumPy
environment active:

```bash
GROK_MODEL_ROOT=/path/to/xai-grok-1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/grok1_alpha_schedule_ablation.py \
  --npy-root "$GROK_MODEL_ROOT/export-npy" \
  --pack-root "$GROK_MODEL_ROOT/artifacts/multiblock-68" \
  --embedding-shard "$GROK_MODEL_ROOT/export-npy/embedding__slot_00__token_embedding.npy" \
  --out reports/grok-1-int4-alpha-schedule-ablation/new-run \
  --preflight-only
```

Use a new output directory to retain this historical preflight. Only after
fresh resource gates and a whole-host competing-workload check pass, use the
same command without `--preflight-only`. Each heavyweight child rechecks its
resource gates. Full input hashes, all four cells and controls, routing/cosine/
residual metrics, paired contrasts and measured resource results remain
outstanding. This header-only preflight does not attest full input identity.
