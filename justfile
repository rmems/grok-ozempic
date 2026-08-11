# grok-ozempic verification surface (#62 / RM-250)
#
# Named tiers for humans and coding agents. Prefer these over rediscovering
# Cargo features, Python test modules, or local checkpoint paths.
#
# Discovery:  just --list
# Pre-push gate: just review  (see REVIEW.md + .githooks/)
# No multi-GiB weights required for: check, test, build, ci, review, doctor
# experiment-smoke needs local CKPT / GROK_OZEMPIC_DISSECT_RUN (fails loud if missing)
# review-full / qodana need the Qodana CLI (and usually Docker)
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
    if ! python3 -c 'import numpy' >/dev/null 2>&1; then
      echo "error: numpy is required for Python script unittests" >&2
      echo "       install: python3 -m pip install --user 'numpy>=1.26,<3'" >&2
      echo "       (matches .github/workflows/python-scripts.yml)" >&2
      exit 1
    fi
    mods=(
      scripts.test_export_grok1_embedding_npy
      scripts.test_export_grok1_int8_npy
      scripts.test_export_grok1_int8_select
      scripts.test_route_preservation_surface
      scripts.test_route_preservation_io
      scripts.test_grok1_multiblock_experiment
    )
    for m in "${mods[@]}"; do
      echo "+ python3 -m unittest ${m} -v"
      python3 -m unittest "${m}" -v
    done

# Block-forward / block-weights unittests (not yet in python-scripts.yml / just ci)
_python-tests-extra:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! python3 -c 'import numpy' >/dev/null 2>&1; then
      echo "error: numpy is required for Python script unittests" >&2
      echo "       install: python3 -m pip install --user 'numpy>=1.26,<3'" >&2
      exit 1
    fi
    mods=(
      scripts.test_grok1_block_forward
      scripts.test_grok1_block_weights
    )
    for m in "${mods[@]}"; do
      echo "+ python3 -m unittest ${m} -v"
      python3 -m unittest "${m}" -v
    done

# cargo-audit.yml parity when the binary is installed; skip (warn) otherwise
_cargo-audit:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v cargo-audit >/dev/null 2>&1; then
      echo "+ cargo audit"
      cargo audit
    elif cargo audit -V >/dev/null 2>&1; then
      echo "+ cargo audit"
      cargo audit
    else
      echo "skip: cargo-audit not installed (cargo install cargo-audit --locked)"
    fi

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

_py-compile:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    scripts=(scripts/*.py)
    if [[ ${#scripts[@]} -eq 0 ]]; then
      echo "warn: no scripts/*.py found"
      exit 0
    fi
    echo "+ python3 -m py_compile ${scripts[*]}"
    python3 -m py_compile "${scripts[@]}"

_codex-hook-tests:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v jq >/dev/null 2>&1; then
      echo "error: jq not on PATH — required for _codex-hook-tests (jq empty .codex/hooks.json)" >&2
      echo "       install: https://jqlang.org/download/ or apt-get -y install jq / HOMEBREW_NO_AUTO_UPDATE=1 brew install jq" >&2
      exit 1
    fi
    echo '+ jq empty .codex/hooks.json'
    jq empty .codex/hooks.json
    echo '+ jq empty .muse/hooks.json'
    jq empty .muse/hooks.json
    for hook in .codex/hooks/*.sh .muse/hooks/*.sh; do
      echo "+ bash -n ${hook}"
      bash -n "${hook}"
    done
    echo '+ bash .codex/hooks/test-coauthor-hooks.sh'
    bash .codex/hooks/test-coauthor-hooks.sh

_optional-linters:
    #!/usr/bin/env bash
    # Fail-fast when an installed optional linter finds issues (just ci must not greenwash).
    set -euo pipefail
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
    @just _codex-hook-tests
    @just _optional-linters

# Pre-push quality gate (REVIEW.md + .githooks/pre-push). No Qodana/Docker/weights.
review:
    @just ci
    @just _cargo-audit
    @just _python-tests-extra
    @just _py-compile

# Local JetBrains Qodana (qodana-rust). Needs `qodana` on PATH; results under .qodana/
qodana:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v qodana >/dev/null 2>&1; then
      echo "error: qodana CLI not on PATH" >&2
      echo "       install: https://www.jetbrains.com/help/qodana/getting-started.html" >&2
      echo "       see REVIEW.md § Local Qodana CLI" >&2
      exit 1
    fi
    mkdir -p .qodana/results .qodana/report
    echo "+ qodana scan --linter qodana-rust --print-problems (results → .qodana/)"
    qodana scan \
      --linter qodana-rust \
      --project-dir . \
      --results-dir .qodana/results \
      --report-dir .qodana/report \
      --print-problems \
      --save-report

# Full pre-push including local Qodana scan (slow; optional before large PRs)
review-full:
    @just review
    @just qodana

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

    # HOME may be unset under set -u; default paths only when present.
    home="${HOME-}"
    if [[ -n "${CKPT-}" ]]; then
      CKPT="${CKPT}"
    elif [[ -n "${home}" ]]; then
      CKPT="${home}/.models/xai-grok-1/ckpt-0"
    else
      CKPT=""
    fi

    # GROK_OZEMPIC_DISSECT_RUN may be either:
    #   (a) run root (.../LATEST_CORRECT_GROK1_RUN) — scripts/block_pilot_goz1.sh style, or
    #   (b) already-resolved run3 dir (.../manifests/xai-grok-1-ckpt-0) — Rust test contract
    #       in stream.rs (joins conversion-manifest.json directly).
    if [[ -n "${GROK_OZEMPIC_DISSECT_RUN-}" ]]; then
      DISSECT_RUN="${GROK_OZEMPIC_DISSECT_RUN}"
    elif [[ -n "${home}" ]]; then
      DISSECT_RUN="${home}/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN"
    else
      DISSECT_RUN=""
    fi
    RUN3=""
    if [[ -n "${DISSECT_RUN}" ]]; then
      if [[ -f "${DISSECT_RUN}/conversion-manifest.json" ]]; then
        RUN3="${DISSECT_RUN}"
      elif [[ -f "${DISSECT_RUN}/manifests/xai-grok-1-ckpt-0/conversion-manifest.json" ]]; then
        RUN3="${DISSECT_RUN}/manifests/xai-grok-1-ckpt-0"
      fi
    fi
    missing=0

    if [[ -z "${CKPT}" || ! -d "${CKPT}" ]]; then
      echo "error: missing checkpoint directory: ${CKPT:-'(unset; set CKPT or HOME)'}" >&2
      echo "       set CKPT to your xai-grok-1 ckpt-0 path" >&2
      missing=1
    else
      echo "ok: CKPT=${CKPT}"
    fi

    if [[ -z "${RUN3}" ]]; then
      echo "error: could not resolve run3 manifests from GROK_OZEMPIC_DISSECT_RUN=${DISSECT_RUN:-'(unset)'}" >&2
      echo "       set GROK_OZEMPIC_DISSECT_RUN to either the xai-dissect run root or the" >&2
      echo "       .../manifests/xai-grok-1-ckpt-0 directory (must contain conversion-manifest.json)" >&2
      missing=1
    else
      for required in conversion-manifest.json quant-plan.json pilot-selection-plan.json; do
        if [[ ! -f "${RUN3}/${required}" ]]; then
          echo "error: missing run3 ${required} under ${RUN3}" >&2
          missing=1
        fi
      done
    fi
    if [[ ${missing} -eq 0 ]]; then
      echo "ok: GROK_OZEMPIC_DISSECT_RUN=${DISSECT_RUN}"
      echo "ok: run3 manifests=${RUN3}"
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
    if command -v jq >/dev/null 2>&1; then
      status ok "jq $(jq --version 2>/dev/null | head -1)"
    else
      status missing "jq not on PATH (required for just ci coauthor hook tests)"
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
    home="${HOME-}"
    if [[ -n "${CKPT-}" ]]; then
      CKPT="${CKPT}"
    elif [[ -n "${home}" ]]; then
      CKPT="${home}/.models/xai-grok-1/ckpt-0"
    else
      CKPT=""
    fi
    if [[ -n "${GROK_OZEMPIC_DISSECT_RUN-}" ]]; then
      DISSECT_RUN="${GROK_OZEMPIC_DISSECT_RUN}"
    elif [[ -n "${home}" ]]; then
      DISSECT_RUN="${home}/rmems/grok-result/xai-dissect/LATEST_CORRECT_GROK1_RUN"
    else
      DISSECT_RUN=""
    fi
    # Accept run root or already-resolved manifests dir (see experiment-smoke).
    RUN3=""
    if [[ -n "${DISSECT_RUN}" ]]; then
      if [[ -f "${DISSECT_RUN}/conversion-manifest.json" ]]; then
        RUN3="${DISSECT_RUN}"
      elif [[ -f "${DISSECT_RUN}/manifests/xai-grok-1-ckpt-0/conversion-manifest.json" ]]; then
        RUN3="${DISSECT_RUN}/manifests/xai-grok-1-ckpt-0"
      fi
    fi
    if [[ -z "${home}" && -z "${CKPT-}" ]]; then
      status warn "HOME unset and CKPT unset — cannot form default checkpoint path"
    elif [[ -n "${CKPT}" && -d "${CKPT}" ]]; then
      status ok "CKPT=${CKPT}"
    else
      status missing "CKPT=${CKPT:-'(unset)'} (set CKPT for experiments)"
    fi
    if [[ -z "${home}" && -z "${GROK_OZEMPIC_DISSECT_RUN-}" ]]; then
      status warn "HOME unset and GROK_OZEMPIC_DISSECT_RUN unset — cannot form default run3 path"
    elif [[ -n "${DISSECT_RUN}" && -d "${DISSECT_RUN}" ]]; then
      status ok "GROK_OZEMPIC_DISSECT_RUN=${DISSECT_RUN}"
    else
      status missing "GROK_OZEMPIC_DISSECT_RUN=${DISSECT_RUN:-'(unset)'}"
    fi
    if [[ -z "${RUN3}" ]]; then
      status missing "run3 manifests (conversion-manifest.json not found under DISSECT_RUN)"
    else
      for required in conversion-manifest.json quant-plan.json pilot-selection-plan.json; do
        if [[ -f "${RUN3}/${required}" ]]; then
          status ok "run3 ${required}"
        else
          status missing "run3 ${RUN3}/${required}"
        fi
      done
    fi

    echo "=== GPU / CUDA (informational) ==="
    if command -v nvidia-smi >/dev/null 2>&1; then
      status ok "nvidia-smi present (CUDA kernels owned by myelin-accelerator; this crate uses LocalBackend CPU)"
    else
      status warn "nvidia-smi not found — fine for this crate (CPU LocalBackend); CUDA is out of scope here"
    fi

    echo "=== doctor complete (exit 0) ==="
    exit 0
