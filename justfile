# grok-ozempic verification surface (#62 / RM-250)
#
# Named tiers for humans and coding agents. Prefer these over rediscovering
# Cargo features, Python test modules, or local checkpoint paths.
#
# Discovery:  just --list
# No multi-GiB weights required for: check, test, build, ci, doctor
# experiment-smoke needs local CKPT / GROK_OZEMPIC_DISSECT_RUN (fails loud if missing)
# Kernel CUDA benches live in myelin-accelerator, not here.

# Default: list all verification tiers.
default:
    @just --list

# ---------------------------------------------------------------------------
# Shared helpers (private — not listed by just --list if prefixed with _)
# ---------------------------------------------------------------------------

# Python unittest modules matching .github/workflows/python-scripts.yml
_python-tests:
    #!/usr/bin/env bash
    set -euo pipefail
    mods=(
      scripts.test_export_grok1_embedding_npy
      scripts.test_export_grok1_int8_npy
      scripts.test_export_grok1_int8_select
      scripts.test_route_preservation_surface
      scripts.test_route_preservation_io
    )
    for m in "${mods[@]}"; do
      echo "+ python3 -m unittest ${m} -v"
      python3 -m unittest "${m}" -v
    done

_bash-n-scripts:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    scripts=(scripts/*.sh)
    if [[ ${#scripts[@]} -eq 0 ]]; then
      echo "warn: no scripts/*.sh found"
      exit 0
    fi
    for f in "${scripts[@]}"; do
      echo "+ bash -n ${f}"
      bash -n "${f}"
    done

_optional-linters:
    #!/usr/bin/env bash
    set -uo pipefail
    if command -v actionlint >/dev/null 2>&1; then
      echo "+ actionlint"
      actionlint
    else
      echo "skip: actionlint not installed"
    fi
    if command -v shellcheck >/dev/null 2>&1; then
      shopt -s nullglob
      scripts=(scripts/*.sh)
      if [[ ${#scripts[@]} -gt 0 ]]; then
        echo "+ shellcheck ${scripts[*]}"
        shellcheck "${scripts[@]}"
      fi
    else
      echo "skip: shellcheck not installed"
    fi

# ---------------------------------------------------------------------------
# Public tiers
# ---------------------------------------------------------------------------

# Fast fmt + clippy (cli features). No weights.
check:
    @echo '+ cargo fmt --all -- --check'
    @cargo fmt --all -- --check
    @echo '+ cargo clippy --all-targets --features cli --locked -- -D warnings'
    @cargo clippy --all-targets --features cli --locked -- -D warnings

# Rust CLI tests + Python unittests. No multi-GiB weights.
test:
    @echo '+ cargo test --features cli --locked'
    @cargo test --features cli --locked
    @just _python-tests

# cargo build --all-targets --all-features --locked
build:
    @echo '+ cargo build --all-targets --all-features --locked'
    @cargo build --all-targets --all-features --locked

# No harness yet (exit 1). Kernel benches → myelin-accelerator.
bench:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "error: no benchmark harness in this crate yet" >&2
    echo "       kernel benches belong in myelin-accelerator" >&2
    echo "       (replace this recipe with \`cargo bench --locked\` once [[bench]] exists)" >&2
    exit 1

# Pre-PR parity: rust.yml + python-scripts.yml (+ optional linters). No Docker/CUDA.
ci:
    @echo '+ cargo fmt --all -- --check'
    @cargo fmt --all -- --check
    @echo '+ cargo clippy --all-targets --all-features --locked -- -D warnings'
    @cargo clippy --all-targets --all-features --locked -- -D warnings
    @echo '+ cargo test --all-targets --all-features --locked'
    @cargo test --all-targets --all-features --locked
    @echo '+ cargo build --all-targets --all-features --locked'
    @cargo build --all-targets --all-features --locked
    @echo '+ cargo doc --no-deps --all-features --locked'
    @cargo doc --no-deps --all-features --locked
    @just _python-tests
    @just _bash-n-scripts
    @just _optional-linters

# Release CLI --help smoke; require CKPT + run3 data (fail loud if missing).
experiment-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    echo '+ cargo build --release --features cli --locked'
    cargo build --release --features cli --locked
    bin=target/release/grok-ozempic
    echo "+ ${bin} quantize-goz1 --help"
    "${bin}" quantize-goz1 --help >/dev/null
    echo "+ ${bin} validate-ingest --help"
    "${bin}" validate-ingest --help >/dev/null
    echo "ok: CLI help smoke passed"

    CKPT="${CKPT:-${HOME}/.models/xai-grok-1/ckpt-0}"
    DISSECT_RUN="${GROK_OZEMPIC_DISSECT_RUN:-${HOME}/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN}"
    RUN3="${DISSECT_RUN}/manifests/xai-grok-1-ckpt-0"
    missing=0

    if [[ ! -d "${CKPT}" ]]; then
      echo "error: missing checkpoint directory: ${CKPT}" >&2
      echo "       set CKPT to your xai-grok-1 ckpt-0 path" >&2
      missing=1
    else
      echo "ok: CKPT=${CKPT}"
    fi

    for required in conversion-manifest.json quant-plan.json pilot-selection-plan.json; do
      if [[ ! -f "${RUN3}/${required}" ]]; then
        echo "error: missing run3 ${required} under ${RUN3}" >&2
        echo "       set GROK_OZEMPIC_DISSECT_RUN to the xai-dissect run root" >&2
        missing=1
      fi
    done
    if [[ ${missing} -eq 0 ]]; then
      echo "ok: GROK_OZEMPIC_DISSECT_RUN=${DISSECT_RUN}"
      echo "ok: experiment data present (not running full pilot; use scripts/block_pilot_goz1.sh)"
    else
      exit 1
    fi

# Env/tool/path diagnosis (ok/warn/missing). Always exits 0.
doctor:
    #!/usr/bin/env bash
    set -uo pipefail

    status() {
      local level="$1" msg="$2"
      printf '%s: %s\n' "${level}" "${msg}"
    }

    echo "=== tools ==="
    if command -v rustc >/dev/null 2>&1; then
      status ok "rustc $(rustc --version 2>/dev/null | head -1)"
    else
      status missing "rustc not on PATH"
    fi
    if command -v cargo >/dev/null 2>&1; then
      status ok "cargo $(cargo --version 2>/dev/null | head -1)"
    else
      status missing "cargo not on PATH"
    fi
    if command -v just >/dev/null 2>&1; then
      status ok "just $(just --version 2>/dev/null | head -1)"
    else
      status missing "just not on PATH"
    fi
    if command -v python3 >/dev/null 2>&1; then
      status ok "python3 $(python3 --version 2>/dev/null)"
    else
      status missing "python3 not on PATH"
    fi
    if python3 -c 'import numpy' >/dev/null 2>&1; then
      status ok "numpy $(python3 -c 'import numpy; print(numpy.__version__)')"
    else
      status warn "numpy not importable (needed for int8 export / route-preservation scripts)"
    fi
    if command -v actionlint >/dev/null 2>&1; then
      status ok "actionlint present"
    else
      status warn "actionlint not installed (optional for just ci)"
    fi
    if command -v shellcheck >/dev/null 2>&1; then
      status ok "shellcheck present"
    else
      status warn "shellcheck not installed (optional for just ci)"
    fi

    echo "=== crate / CLI ==="
    if [[ -f Cargo.toml ]]; then
      status ok "Cargo.toml present"
    else
      status missing "Cargo.toml (run from repo root)"
    fi
    if [[ -x target/release/grok-ozempic ]]; then
      status ok "release binary target/release/grok-ozempic"
    elif [[ -x target/debug/grok-ozempic ]]; then
      status ok "debug binary target/debug/grok-ozempic"
    else
      status warn "no grok-ozempic binary built yet (cargo build --features cli)"
    fi
    if [[ -f dissect/grok-1/structural-manifest.json ]]; then
      status ok "dissect/grok-1/structural-manifest.json"
    else
      status missing "dissect/grok-1/structural-manifest.json"
    fi
    if [[ -f dissect/grok-1/baseline.json ]]; then
      status ok "dissect/grok-1/baseline.json"
    else
      status missing "dissect/grok-1/baseline.json"
    fi

    echo "=== local experiment data (optional) ==="
    CKPT="${CKPT:-${HOME}/.models/xai-grok-1/ckpt-0}"
    DISSECT_RUN="${GROK_OZEMPIC_DISSECT_RUN:-${HOME}/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN}"
    RUN3="${DISSECT_RUN}/manifests/xai-grok-1-ckpt-0"
    if [[ -d "${CKPT}" ]]; then
      status ok "CKPT=${CKPT}"
    else
      status missing "CKPT=${CKPT} (set CKPT for experiments)"
    fi
    if [[ -d "${DISSECT_RUN}" ]]; then
      status ok "GROK_OZEMPIC_DISSECT_RUN=${DISSECT_RUN}"
    else
      status missing "GROK_OZEMPIC_DISSECT_RUN=${DISSECT_RUN}"
    fi
    for required in conversion-manifest.json quant-plan.json pilot-selection-plan.json; do
      if [[ -f "${RUN3}/${required}" ]]; then
        status ok "run3 ${required}"
      else
        status missing "run3 ${RUN3}/${required}"
      fi
    done

    echo "=== GPU / CUDA (informational) ==="
    if command -v nvidia-smi >/dev/null 2>&1; then
      status ok "nvidia-smi present (CUDA kernels owned by myelin-accelerator; this crate uses LocalBackend CPU)"
    else
      status warn "nvidia-smi not found — fine for this crate (CPU LocalBackend); CUDA is out of scope here"
    fi

    echo "=== doctor complete (exit 0) ==="
    exit 0
