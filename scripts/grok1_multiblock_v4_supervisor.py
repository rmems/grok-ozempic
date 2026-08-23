#!/usr/bin/env python3
"""Supervise the locked GH #85 three-arm real-weight experiment.

The numerical experiment imports NumPy and can be killed before Python gets a
chance to handle ``MemoryError``.  This module intentionally uses only the
standard library and runs each numerical arm in a separate child process.  The
parent survives a child SIGKILL, preserves the last atomic progress record, and
publishes a canonical fail-closed Option-4 artifact.

The recipe is deliberately not configurable: 8192 tokens, seed 20260806,
blocks 0,1,2,3, top-k 2, and the FP16 control are required.  In particular,
there is no 2048-token retry or fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import shlex
import signal

# Fixed interpreter/script argv; the call below always disables the shell.
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXIT_OK = 0
# grok1_multiblock_experiment uses 1, 4, and 5; block0 also uses 3.
EXIT_SUPERVISOR_FAILCLOSED = 6

TOKENS = 8192
SEED = 20260806
BLOCKS = (0, 1, 2, 3)
BLOCKS_ARG = "0,1,2,3"
TOP_K = 2
PROTOCOL = {
    "tokens": TOKENS,
    "seed": SEED,
    "blocks": list(BLOCKS),
    "top_k": TOP_K,
    "fp16_control": True,
    "fallback_tokens": None,
}

AGENT_LINE = "Grok Build: Grok 4.5 (xAI) · Issue: #85 / Linear RM-608 · beads goz-3h3"
ISSUE = "GH #85 / Linear RM-608 / beads goz-3h3"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXPERIMENT_SCRIPT = SCRIPT_DIR / "grok1_multiblock_experiment.py"
DEFAULT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPORT_OPTION_RE = re.compile(
    r"(?m)^(?:\*\*Decision:\*\*\s*Option\s+(\d+)|\*\*Option\s+(\d+)\s+[—-])"
)


class ArtifactError(RuntimeError):
    """A zero-exit child did not produce complete canonical evidence."""

    def __init__(self, failure_class: str, errors: Sequence[str], payload: Any = None):
        super().__init__("; ".join(errors))
        self.failure_class = failure_class
        self.errors = list(errors)
        self.payload = payload


@dataclass(frozen=True)
class ArmSpec:
    stage: str
    expected_label: str
    output_name: str | None
    progress_name: str
    cli_tail: tuple[str, ...]
    evidence_only: bool


ARMS = (
    ArmSpec(
        stage="baseline",
        expected_label="expert_int4",
        output_name="int4-baseline",
        progress_name="progress-int4-baseline.json",
        cli_tail=("--arm", "int4", "--evidence-only"),
        evidence_only=True,
    ),
    ArmSpec(
        stage="p1",
        expected_label="expert_int4_channel_alpha_123",
        output_name="int4-channel-alpha-123",
        progress_name="progress-int4-channel-alpha-123.json",
        cli_tail=(
            "--arm",
            "int4_channel_alpha",
            "--hp-blocks",
            "1,2,3",
            "--evidence-only",
        ),
        evidence_only=True,
    ),
    ArmSpec(
        stage="p0",
        expected_label="expert_int4_channel_alpha",
        output_name=None,
        progress_name="progress-int4-channel-alpha.json",
        cli_tail=("--arm", "int4_channel_alpha", "--write-report-md"),
        evidence_only=False,
    ),
)

_ARM_BY_STAGE = {arm.stage: arm for arm in ARMS}
_REQUIRED_LABELS = tuple(arm.expected_label for arm in ARMS)
_CANDIDATE_LABELS = (
    _ARM_BY_STAGE["p0"].expected_label,
    _ARM_BY_STAGE["p1"].expected_label,
)
_LABEL_BY_ALIAS = {
    **{arm.stage: arm.expected_label for arm in ARMS},
    **{arm.expected_label: arm.expected_label for arm in ARMS},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, body: str) -> None:
    """Durably replace one small text artifact from a same-directory temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
        # Best effort directory fsync makes the rename durable on filesystems
        # that support it.  The final file is already atomic if this fails.
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _best_effort_write_json(path: Path, payload: object) -> None:
    """Write a supplemental diagnostic without masking canonical publication."""
    try:
        _atomic_write_json(path, payload)
    except OSError as exc:
        print(f"warning: could not write supplemental diagnostic {path}: {exc}", file=sys.stderr)


def _exact_int(value: object, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _exact_int_list(value: object, expected: Sequence[int]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    return all(_exact_int(item, want) for item, want in zip(value, expected))


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _artifact_signature(path: Path) -> tuple[int, int, int, int] | None:
    """Return enough stat identity to reject a stale zero-exit artifact."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _arm_output(out: Path, arm: ArmSpec) -> Path:
    return out if arm.output_name is None else out / arm.output_name


def _arm_progress(out: Path, arm: ArmSpec) -> Path:
    return out / arm.progress_name


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError("invalid_evidence", [f"missing artifact: {path}"]) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            "invalid_evidence", [f"unreadable or malformed artifact {path}: {exc}"]
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactError(
            "invalid_evidence", [f"artifact {path} is not a JSON object"]
        )
    return value


def _schedule_for_arm(arm: ArmSpec) -> tuple[str, list[int], list[int], list[int]]:
    if arm.stage == "baseline":
        return "int4", [], list(BLOCKS), []
    if arm.stage == "p1":
        return "int4_channel_alpha", [1, 2, 3], [0], [0]
    return "int4_channel_alpha", [], list(BLOCKS), list(BLOCKS)


def _protocol_errors(chain: object, arm: ArmSpec) -> list[str]:
    if not isinstance(chain, dict):
        return ["missing chain object"]
    errors: list[str] = []
    if chain.get("arm_label") != arm.expected_label:
        errors.append(
            f"arm_label={chain.get('arm_label')!r} expected={arm.expected_label!r}"
        )
    if not _exact_int(chain.get("tokens"), TOKENS):
        errors.append(f"tokens={chain.get('tokens')!r} expected={TOKENS}")
    if not _exact_int(chain.get("token_seed"), SEED):
        errors.append(f"token_seed={chain.get('token_seed')!r} expected={SEED}")
    if not _exact_int(chain.get("top_k"), TOP_K):
        errors.append(f"top_k={chain.get('top_k')!r} expected={TOP_K}")
    if not _exact_int_list(chain.get("blocks"), BLOCKS):
        errors.append(f"blocks={chain.get('blocks')!r} expected={list(BLOCKS)!r}")
    if chain.get("skip_fp16_control") is not False:
        errors.append("FP16 control was skipped or not explicitly recorded")

    mode, hp_blocks, int4_blocks, channel_blocks = _schedule_for_arm(arm)
    for field, expected in (
        ("hp_blocks", hp_blocks),
        ("int4_blocks", int4_blocks),
        ("channel_alpha_blocks", channel_blocks),
    ):
        if not _exact_int_list(chain.get(field), expected):
            errors.append(f"{field}={chain.get(field)!r} expected={expected!r}")
    if chain.get("expert_mode") != mode:
        errors.append(f"expert_mode={chain.get('expert_mode')!r} expected={mode!r}")

    rows = chain.get("per_block")
    if not isinstance(rows, list) or len(rows) != len(BLOCKS):
        errors.append("per_block must contain exactly the four locked blocks")
        return errors
    observed: list[object] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"per_block[{index}] is not an object")
            continue
        observed.append(row.get("block"))
        if not isinstance(row.get("expert_only"), dict):
            errors.append(f"per_block[{index}] missing expert_only metrics")
        control = row.get("fp16_control")
        if not isinstance(control, dict):
            errors.append(f"per_block[{index}] missing FP16 control metrics")
            continue
        cosine = control.get("block_output_cosine")
        if (
            isinstance(cosine, bool)
            or not isinstance(cosine, (int, float))
            or not math.isfinite(float(cosine))
            or float(cosine) < 0.99
            or float(cosine) > 1.0 + 1e-9
        ):
            errors.append(f"per_block[{index}] FP16 control cosine is not clean")
    if not _exact_int_list(observed, BLOCKS):
        errors.append(f"per_block order={observed!r} expected={list(BLOCKS)!r}")
    return errors


def _provenance_errors(payload: dict[str, Any], arm: ArmSpec) -> list[str]:
    errors: list[str] = []
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return ["missing provenance object"]
    implementation = provenance.get("implementation")
    if not isinstance(implementation, dict) or not implementation:
        errors.append("missing implementation provenance")
    if not isinstance(provenance.get("model"), str) or not provenance.get("model"):
        errors.append("missing model provenance")
    if not isinstance(provenance.get("embedding_shard"), str) or not provenance.get(
        "embedding_shard"
    ):
        errors.append("missing embedding input identity")
    if provenance.get("skip_fp16_control") is not False:
        errors.append("provenance does not confirm FP16 control")
    expected_cli_arm = "int4" if arm.stage == "baseline" else "int4_channel_alpha"
    if provenance.get("arm") != expected_cli_arm:
        errors.append(
            f"provenance arm={provenance.get('arm')!r} expected={expected_cli_arm!r}"
        )
    role = provenance.get("evidence_role")
    if arm.evidence_only:
        if role != "secondary; no independent decision":
            errors.append("secondary evidence role is missing or invalid")
    elif not isinstance(role, str) or "primary" not in role or "#85" not in role:
        errors.append("primary #85 evidence role is missing or invalid")
    return errors


def _pack_provenance_errors(chain: object, arm: ArmSpec) -> list[str]:
    if not isinstance(chain, dict):
        return []
    rows = chain.get("pack_provenance")
    if not isinstance(rows, list) or len(rows) != len(BLOCKS):
        return ["pack_provenance must contain exactly the four locked blocks"]
    errors: list[str] = []
    expected_sources = {
        block: (
            "fp16_control"
            if arm.stage == "p1" and block in (1, 2, 3)
            else (
                "research_int4_side"
                if arm.stage == "baseline"
                else "research_int4_channel_alpha_side"
            )
        )
        for block in BLOCKS
    }
    observed: list[object] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"pack_provenance[{index}] is not an object")
            continue
        block = row.get("block")
        observed.append(block)
        for field in ("pack_sha256", "npy_sha256"):
            if not _sha256(row.get(field)):
                errors.append(f"pack_provenance[{index}] has invalid {field}")
        sources = row.get("scale_sources")
        if not isinstance(sources, dict) or not sources:
            errors.append(f"pack_provenance[{index}] missing applied scale sources")
        elif _exact_int(block, index):
            values = set(sources.values())
            expected = expected_sources[index]
            if values != {expected}:
                errors.append(
                    f"pack_provenance[{index}] scale sources={sorted(map(str, values))!r} "
                    f"expected={expected!r}"
                )
    if not _exact_int_list(observed, BLOCKS):
        errors.append(f"pack_provenance order={observed!r} expected={list(BLOCKS)!r}")
    return errors


def _normalized_arm_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in _LABEL_BY_ALIAS:
            return None
        normalized.append(_LABEL_BY_ALIAS[item])
    return normalized


def _primary_decision_errors(payload: dict[str, Any]) -> tuple[list[str], int | None]:
    errors: list[str] = []
    decision = payload.get("decision")
    raw_option = decision.get("decision") if isinstance(decision, dict) else None
    if (
        isinstance(raw_option, bool)
        or not isinstance(raw_option, int)
        or raw_option not in (1, 2, 3)
    ):
        errors.append("primary decision must be exactly one of options 1, 2, or 3")
        decision_option = None
    else:
        decision_option = raw_option
        winner = decision.get("best_remedy_arm")
        if winner not in _CANDIDATE_LABELS:
            errors.append(
                "primary decision winner is missing or is not a remedy candidate"
            )
        if decision.get("protocol_complete") is not True:
            errors.append("primary decision does not confirm protocol_complete")
    return errors, decision_option


def _comparison_completeness_errors(comparison: object) -> list[str]:
    if not isinstance(comparison, dict):
        return ["primary payload missing comparison object"]
    errors: list[str] = []
    if comparison.get("protocol_complete") is not True:
        errors.append("comparison.protocol_complete is not true")
    completed = _normalized_arm_list(comparison.get("completed_arms"))
    if (
        completed is None
        or set(completed) != set(_REQUIRED_LABELS)
        or len(completed) != len(_REQUIRED_LABELS)
    ):
        errors.append(
            f"comparison.completed_arms={comparison.get('completed_arms')!r} "
            f"does not cover {list(_REQUIRED_LABELS)!r}"
        )
    if comparison.get("missing_arms") != []:
        errors.append("comparison reports missing arms")
    if comparison.get("invalid_arms") != []:
        errors.append("comparison reports invalid arms")
    if comparison.get("validation_errors") not in (None, []):
        errors.append("comparison contains validation errors")
    return errors


def _ranking_errors(payload: dict[str, Any]) -> list[str]:
    comparison = payload.get("comparison")
    decision = payload.get("decision")
    if not isinstance(comparison, dict) or not isinstance(decision, dict):
        return []
    ranking = comparison.get("ranking")
    if not isinstance(ranking, dict):
        return ["complete comparison is missing the canonical ranking contract"]
    errors: list[str] = []
    ordered = _normalized_arm_list(ranking.get("ordered_candidates"))
    if (
        ordered is None
        or set(ordered) != set(_CANDIDATE_LABELS)
        or len(ordered) != len(_CANDIDATE_LABELS)
    ):
        errors.append("ranking must contain exactly P0 and P1, never the INT4 baseline")
    winner = ranking.get("winner")
    if winner not in _CANDIDATE_LABELS:
        errors.append("ranking winner is not P0 or P1")
    if ranking.get("baseline_comparator") != _ARM_BY_STAGE["baseline"].expected_label:
        errors.append(
            "ranking baseline comparator is not the re-measured plain INT4 arm"
        )
    deltas = ranking.get("baseline_deltas")
    if not isinstance(deltas, dict) or set(deltas) != set(_CANDIDATE_LABELS):
        errors.append("ranking baseline deltas do not cover exactly P0 and P1")
    tie_break = ranking.get("tie_break_reason")
    if not isinstance(tie_break, str) or not tie_break:
        errors.append("ranking tie-break reason is missing")
    if comparison.get("best_remedy_arm") != winner:
        errors.append("comparison winner differs from ranking winner")
    if decision.get("best_remedy_arm") != winner:
        errors.append("decision winner differs from ranking winner")
    if decision.get("ordered_candidates") != ranking.get("ordered_candidates"):
        errors.append("decision ordered candidates differ from ranking")
    if decision.get("baseline_deltas") != deltas:
        errors.append("decision baseline deltas differ from ranking")
    if decision.get("tie_break_reason") != tie_break:
        errors.append("decision tie-break reason differs from ranking")
    return errors


def _report_errors(out: Path, decision_option: int | None) -> list[str]:
    report = out / "results.md"
    try:
        body = report.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"canonical results.md is missing or unreadable: {exc}"]
    match = _REPORT_OPTION_RE.search(body)
    report_option = int(match.group(1) or match.group(2)) if match else None
    if decision_option is not None and report_option != decision_option:
        return [
            f"results.md option={report_option!r} does not match metrics option={decision_option}"
        ]
    return []


def _primary_errors(payload: dict[str, Any], out: Path) -> list[str]:
    errors, decision_option = _primary_decision_errors(payload)
    errors.extend(_comparison_completeness_errors(payload.get("comparison")))
    errors.extend(_ranking_errors(payload))
    errors.extend(_report_errors(out, decision_option))
    return errors


def _validate_artifact(
    metrics_path: Path,
    arm: ArmSpec,
    *,
    signature_before: tuple[int, int, int, int] | None,
    canonical_out: Path,
) -> dict[str, Any]:
    signature_after = _artifact_signature(metrics_path)
    if signature_after is None:
        raise ArtifactError("invalid_evidence", [f"missing artifact: {metrics_path}"])
    if signature_before is not None and signature_after == signature_before:
        raise ArtifactError(
            "invalid_evidence",
            [f"child did not refresh stale artifact: {metrics_path}"],
        )
    payload = _read_json_object(metrics_path)
    protocol_errors = _protocol_errors(payload.get("chain"), arm)
    if protocol_errors:
        raise ArtifactError("protocol_mismatch", protocol_errors, payload)
    errors = _provenance_errors(payload, arm)
    errors.extend(_pack_provenance_errors(payload.get("chain"), arm))
    if arm.evidence_only:
        if "decision" in payload:
            errors.append("evidence-only artifact must not contain a decision")
    else:
        errors.extend(_primary_errors(payload, canonical_out))
    if errors:
        raise ArtifactError("invalid_evidence", errors, payload)
    return payload


def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    chain = payload.get("chain")
    if not isinstance(provenance, dict) or not isinstance(chain, dict):
        return {}
    packs = chain.get("pack_provenance")
    pack_sha: dict[str, str] = {}
    npy_sha: dict[str, str] = {}
    if isinstance(packs, list):
        for row in packs:
            if not isinstance(row, dict) or not isinstance(row.get("block"), int):
                continue
            key = str(row["block"])
            if _sha256(row.get("pack_sha256")):
                pack_sha[key] = row["pack_sha256"]
            if _sha256(row.get("npy_sha256")):
                npy_sha[key] = row["npy_sha256"]
    return {
        "implementation": provenance.get("implementation"),
        "model": provenance.get("model"),
        "architecture_source": provenance.get("architecture_source"),
        "embedding_shard": provenance.get("embedding_shard"),
        "pack_sha256": pack_sha,
        "npy_sha256": npy_sha,
    }


def _identity_mismatches(
    reference: dict[str, Any], current: dict[str, Any], arm: ArmSpec
) -> list[str]:
    errors: list[str] = []
    for field in (
        "implementation",
        "model",
        "architecture_source",
        "embedding_shard",
        "pack_sha256",
        "npy_sha256",
    ):
        if current.get(field) != reference.get(field):
            errors.append(f"{arm.expected_label}:{field}_mismatch")
    return errors


def _read_progress(
    path: Path, fallback: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return dict(fallback), str(exc)
    if not isinstance(value, dict):
        return dict(fallback), "progress JSON is not an object"
    return value, None


def _progress_detail(progress: dict[str, Any]) -> dict[str, Any]:
    raw_completed = progress.get("completed_blocks")
    completed = (
        [
            item
            for item in raw_completed
            if any(_exact_int(item, block) for block in BLOCKS)
        ]
        if isinstance(raw_completed, list)
        else []
    )
    current = progress.get("current_block")
    if not any(_exact_int(current, block) for block in BLOCKS):
        current = None
    protocol = progress.get("protocol")
    active_tokens = protocol.get("tokens") if isinstance(protocol, dict) else TOKENS
    if not _exact_int(active_tokens, TOKENS):
        active_tokens = TOKENS
    return {
        "current_block": current,
        "completed_blocks": completed,
        "highest_completed_block": max(completed) if completed else None,
        "active_tokens": active_tokens,
    }


def _proc_meminfo() -> dict[str, Any]:
    captured = _utc_now()
    wanted = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    values: dict[str, int] = {}
    error: str | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, separator, raw = line.partition(":")
            if separator and key in wanted:
                parts = raw.split()
                if parts:
                    values[f"{key}_bytes"] = int(parts[0]) * 1024
    except (OSError, UnicodeError, ValueError) as exc:
        error = str(exc)
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        max_rss_kib = int(usage.ru_maxrss)
        children_max_rss_kib = int(child_usage.ru_maxrss)
    except (OSError, ValueError) as exc:
        max_rss_kib = 0
        children_max_rss_kib = 0
        error = f"{error}; {exc}" if error else str(exc)
    result: dict[str, Any] = {
        "captured_at": captured,
        "source": "/proc/meminfo + resource.getrusage",
        "values": values,
        "supervisor_max_rss_kib": max_rss_kib,
        "children_max_rss_kib": children_max_rss_kib,
        "launch_gate_applied": False,
    }
    if error:
        result["error"] = error
    return result


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_child_log(
    path: Path,
    command: Sequence[str],
    *,
    started_at: str,
    ended_at: str,
    returncode: int | None,
    stdout: object,
    stderr: object,
    timeout: bool = False,
) -> None:
    body = "\n".join(
        [
            f"started_at: {started_at}",
            f"ended_at: {ended_at}",
            f"command: {shlex.join(list(command))}",
            f"returncode: {returncode}",
            f"timeout: {str(timeout).lower()}",
            "",
            "----- stdout -----",
            _as_text(stdout),
            "----- stderr -----",
            _as_text(stderr),
        ]
    )
    _atomic_write_text(path, body.rstrip() + "\n")


def _common_child_args(
    args: argparse.Namespace, child_out: Path, progress: Path
) -> list[str]:
    return [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "--npy-root",
        str(args.npy_root),
        "--npy-pattern",
        args.npy_pattern,
        "--pack-root",
        str(args.pack_root),
        "--pack-pattern",
        args.pack_pattern,
        "--embedding-shard",
        str(args.embedding_shard),
        "--int4-side-root",
        str(args.int4_side_root),
        "--tokens",
        str(TOKENS),
        "--seed",
        str(SEED),
        "--blocks",
        BLOCKS_ARG,
        "--top-k",
        str(TOP_K),
        "--out",
        str(child_out),
        "--progress-json",
        str(progress),
    ]


def _child_command(args: argparse.Namespace, arm: ArmSpec) -> list[str]:
    child_out = _arm_output(args.out, arm)
    command = _common_child_args(args, child_out, _arm_progress(args.out, arm))
    command.extend(arm.cli_tail)
    if arm.stage == "p0":
        for evidence in (
            args.out / "int4-baseline" / "metrics.json",
            args.out / "int4-channel-alpha-123" / "metrics.json",
        ):
            command.extend(("--comparison-metrics", str(evidence)))
    return command


def _prelaunch_progress(
    args: argparse.Namespace,
    arm: ArmSpec,
    *,
    supervisor_started_at: str,
    stage_started_at: str,
    completed: list[str],
    start_memory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "prelaunch",
        "stage": arm.stage,
        "arm": arm.cli_tail[1],
        "expected_arm_label": arm.expected_label,
        "planned_arms": [item.expected_label for item in ARMS],
        "completed_arms": list(completed),
        "protocol": dict(PROTOCOL),
        "current_block": None,
        "completed_blocks": [],
        "supervisor_started_at": supervisor_started_at,
        "stage_started_at": stage_started_at,
        "host_memory_at_start": start_memory,
        "input_identity": {
            "npy_root": str(args.npy_root),
            "npy_pattern": args.npy_pattern,
            "pack_root": str(args.pack_root),
            "pack_pattern": args.pack_pattern,
            "embedding_shard": str(args.embedding_shard),
            "int4_side_root": str(args.int4_side_root),
        },
    }


def _signal_fields(returncode: int | None) -> tuple[str | None, int | None]:
    if returncode in (-signal.SIGKILL, 128 + signal.SIGKILL):
        return "SIGKILL", int(signal.SIGKILL)
    if isinstance(returncode, int) and returncode < 0:
        number = -returncode
        try:
            return signal.Signals(number).name, number
        except ValueError:
            return f"SIGNAL_{number}", number
    return None, None


def _failure_text(failure_class: str, arm: ArmSpec, returncode: int | None) -> str:
    if failure_class == "sigkill":
        return (
            f"{arm.expected_label} ended with return code {returncode}; this is consistent "
            "with OOM or an external SIGKILL, but is not authoritative proof of OOM."
        )
    if failure_class == "timeout":
        return f"{arm.expected_label} exceeded the configured child timeout."
    if failure_class == "child_failure":
        return f"{arm.expected_label} exited nonzero with return code {returncode}."
    if failure_class == "launch_error":
        return f"{arm.expected_label} could not be launched."
    if failure_class == "provenance_mismatch":
        return f"{arm.expected_label} does not match prior-arm provenance/identity."
    if failure_class == "protocol_mismatch":
        return f"{arm.expected_label} does not satisfy the locked #85 protocol."
    return f"{arm.expected_label} did not produce valid complete evidence."


def _provenance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    summary: dict[str, Any] = _identity(payload)
    if isinstance(provenance, dict):
        summary.update(
            {
                "issue": provenance.get("issue"),
                "agent": provenance.get("agent"),
                "evidence_role": provenance.get("evidence_role"),
            }
        )
    return summary


def _failure_payload(
    *,
    arm: ArmSpec,
    failure_class: str,
    detail_errors: Sequence[str],
    returncode: int | None,
    progress: dict[str, Any],
    progress_error: str | None,
    completed_specs: list[ArmSpec],
    completed_payloads: dict[str, dict[str, Any]],
    failed_payload: dict[str, Any] | None,
    started_at: str,
    stage_started_at: str,
    failed_at: str,
    start_memory: dict[str, Any],
) -> dict[str, Any]:
    signal_name, signal_number = _signal_fields(returncode)
    progress_detail = _progress_detail(progress)
    completed_labels = [item.expected_label for item in completed_specs]
    failed_index = next(
        index for index, item in enumerate(ARMS) if item.stage == arm.stage
    )
    missing_labels = [item.expected_label for item in ARMS[failed_index + 1 :]]
    invalid_labels = [arm.expected_label]
    decision_text = "Inconclusive — supervised #85 protocol did not complete."
    failure_text = _failure_text(failure_class, arm, returncode)
    rationale = [
        f"failure_class={failure_class}",
        f"failed_arm={arm.expected_label}",
        failure_text,
        f"completed_arms={completed_labels}",
        f"missing_arms={missing_labels}",
        f"completed_blocks={progress_detail['completed_blocks']}",
        "tokens remain locked to 8192; no 2048 fallback was attempted",
        *list(detail_errors),
    ]
    if progress_error:
        rationale.append(f"progress_read_error={progress_error}")
    captured = {
        spec.expected_label: _provenance_summary(completed_payloads[spec.stage])
        for spec in completed_specs
        if spec.stage in completed_payloads
    }
    failed_progress_provenance = {
        "implementation": progress.get("implementation"),
        "input_identity": progress.get("input_identity"),
        "provenance_identity": progress.get("provenance_identity"),
    }
    failed_memory = _proc_meminfo()
    failure = {
        "failure_class": failure_class,
        "failed_stage": arm.stage,
        "failed_arm": arm.expected_label,
        "returncode": returncode,
        "signal": signal_name,
        "signal_number": signal_number,
        "description": failure_text,
        "errors": list(detail_errors),
    }
    return {
        "provenance": {
            "issue": ISSUE,
            "agent": AGENT_LINE,
            "supervisor": Path(__file__).name,
            "python": sys.version.split()[0],
            "implementation": progress.get("implementation")
            or next(
                (
                    _identity(completed_payloads[item.stage]).get("implementation")
                    for item in reversed(completed_specs)
                    if item.stage in completed_payloads
                ),
                None,
            ),
            "completed_arm_provenance": captured,
            "failed_arm_progress_provenance": failed_progress_provenance,
            "failed_arm_artifact_provenance": (
                _provenance_summary(failed_payload)
                if failed_payload is not None
                else None
            ),
        },
        "protocol": dict(PROTOCOL),
        "protocol_complete": False,
        "completed_arms": completed_labels,
        "missing_arms": missing_labels,
        "invalid_arms": invalid_labels,
        "failure_class": failure_class,
        "failed_arm": arm.expected_label,
        "returncode": returncode,
        "signal": signal_name,
        "signal_number": signal_number,
        "current_block": progress_detail["current_block"],
        "completed_blocks": progress_detail["completed_blocks"],
        "highest_completed_block": progress_detail["highest_completed_block"],
        "active_tokens": progress_detail["active_tokens"],
        "started_at": started_at,
        "stage_started_at": stage_started_at,
        "ended_at": failed_at,
        "timestamps": {
            "supervisor_started_at": started_at,
            "stage_started_at": stage_started_at,
            "failed_at": failed_at,
        },
        "host_memory_snapshot": {"start": start_memory, "failure": failed_memory},
        "progress": {
            **progress_detail,
            "child": progress,
            "read_error": progress_error,
        },
        "failure": failure,
        "comparison": {
            "protocol_complete": False,
            "completed_arms": completed_labels,
            "missing_arms": missing_labels,
            "invalid_arms": invalid_labels,
            "validation_errors": list(detail_errors),
        },
        "decision": {
            "decision": 4,
            "decision_text": decision_text,
            "rationale": rationale,
            "compounding": "unknown",
        },
    }


def _render_failure_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    failure = payload["failure"]
    progress = payload["progress"]
    lines = [
        "# Grok-1 expert precision remedy v4",
        "",
        f"**Agent:** {AGENT_LINE}",
        f"**Issue:** {ISSUE}",
        "",
        "## Decision",
        "",
        f"**Option 4 — {decision['decision_text']}**",
        "",
        "The supervised 8192-token protocol is incomplete. No candidate ranking was inferred.",
        "",
        "Rationale:",
        "",
        *[f"- `{item}`" for item in decision["rationale"]],
        "",
        "## Failure classification",
        "",
        f"- Class: `{failure['failure_class']}`",
        f"- Failed arm: `{failure['failed_arm']}`",
        f"- Return code: `{failure['returncode']}`",
        f"- Signal: `{failure['signal']}`",
        f"- Description: {failure['description']}",
        "",
        "## Arm completeness",
        "",
        f"- Completed: `{payload['completed_arms']}`",
        f"- Missing: `{payload['missing_arms']}`",
        f"- Invalid: `{payload['invalid_arms']}`",
        "",
        "## Last atomic progress",
        "",
        f"- Current block: `{progress['current_block']}`",
        f"- Completed blocks: `{progress['completed_blocks']}`",
        f"- Highest completed block: `{progress['highest_completed_block']}`",
        f"- Active tokens: `{progress['active_tokens']}`",
        "",
        "## Locked protocol",
        "",
        f"- Tokens: **{TOKENS}** (no 2048 fallback)",
        f"- Seed: `{SEED}`",
        f"- Blocks: `{BLOCKS_ARG}`",
        f"- Top-k: `{TOP_K}`",
        "- FP16 control: required",
        "",
        "## Host-memory snapshot",
        "",
        "```json",
        json.dumps(payload["host_memory_snapshot"], indent=2, sort_keys=True),
        "```",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(payload["provenance"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _publish_failure(out: Path, payload: dict[str, Any]) -> None:
    """Atomically publish each canonical Option-4 artifact.

    ``metrics.json`` is the machine source of truth and is replaced last, after
    the Markdown report is already durable.
    """
    _atomic_write_text(out / "results.md", _render_failure_report(payload))
    _atomic_write_json(out / "metrics.json", payload)


def _completed_supervisor_progress(
    *,
    status: str,
    started_at: str,
    arm: ArmSpec,
    completed: list[ArmSpec],
) -> dict[str, Any]:
    return {
        "status": status,
        "stage": arm.stage,
        "expected_arm_label": arm.expected_label,
        "protocol": dict(PROTOCOL),
        "completed_arms": [item.expected_label for item in completed],
        "planned_arms": [item.expected_label for item in ARMS],
        "supervisor_started_at": started_at,
        "updated_at": _utc_now(),
    }


def _fail(
    args: argparse.Namespace,
    *,
    arm: ArmSpec,
    failure_class: str,
    errors: Sequence[str],
    returncode: int | None,
    prelaunch: dict[str, Any],
    completed: list[ArmSpec],
    payloads: dict[str, dict[str, Any]],
    failed_payload: dict[str, Any] | None = None,
    started_at: str,
    stage_started_at: str,
    start_memory: dict[str, Any],
) -> int:
    progress, progress_error = _read_progress(_arm_progress(args.out, arm), prelaunch)
    failed_at = _utc_now()
    payload = _failure_payload(
        arm=arm,
        failure_class=failure_class,
        detail_errors=errors,
        returncode=returncode,
        progress=progress,
        progress_error=progress_error,
        completed_specs=completed,
        completed_payloads=payloads,
        failed_payload=failed_payload,
        started_at=started_at,
        stage_started_at=stage_started_at,
        failed_at=failed_at,
        start_memory=start_memory,
    )
    _publish_failure(args.out, payload)
    _best_effort_write_json(
        args.out / "host-at-end.json",
        {
            "status": "failed",
            "failed_arm": arm.expected_label,
            "failure_class": failure_class,
            "memory": payload["host_memory_snapshot"]["failure"],
        },
    )
    _atomic_write_json(
        args.out / "supervisor-progress.json",
        {
            **_completed_supervisor_progress(
                status="failed", started_at=started_at, arm=arm, completed=completed
            ),
            "failure_class": failure_class,
            "failed_arm": arm.expected_label,
            "ended_at": failed_at,
        },
    )
    print(
        f"FAIL-CLOSED option 4: {arm.expected_label}: {_failure_text(failure_class, arm, returncode)}",
        file=sys.stderr,
    )
    return EXIT_SUPERVISOR_FAILCLOSED


def run(args: argparse.Namespace) -> int:
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    args.out = args.out.expanduser()
    args.npy_root = args.npy_root.expanduser()
    args.pack_root = args.pack_root.expanduser()
    args.embedding_shard = args.embedding_shard.expanduser()
    args.int4_side_root = args.int4_side_root.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    start_memory = _proc_meminfo()
    _best_effort_write_json(
        args.out / "host-at-launch.json",
        {
            "status": "prelaunch",
            "protocol": dict(PROTOCOL),
            "memory": start_memory,
        },
    )
    completed: list[ArmSpec] = []
    payloads: dict[str, dict[str, Any]] = {}
    reference_identity: dict[str, Any] | None = None

    for index, arm in enumerate(ARMS, start=1):
        child_out = _arm_output(args.out, arm)
        metrics_path = child_out / "metrics.json"
        signature_before = _artifact_signature(metrics_path)
        stage_started_at = _utc_now()
        prelaunch = _prelaunch_progress(
            args,
            arm,
            supervisor_started_at=started_at,
            stage_started_at=stage_started_at,
            completed=[item.expected_label for item in completed],
            start_memory=start_memory,
        )
        _atomic_write_json(_arm_progress(args.out, arm), prelaunch)
        _atomic_write_json(args.out / "supervisor-progress.json", prelaunch)

        command = _child_command(args, arm)
        print(f"[{index}/3] launching {arm.expected_label}", flush=True)
        try:
            result = subprocess.run(  # nosec B603  # noqa: S603
                command,
                shell=False,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
                cwd=str(REPO_ROOT),
            )
        except subprocess.TimeoutExpired as exc:
            ended_at = _utc_now()
            _write_child_log(
                args.out / f"run-{index:02d}-{arm.stage}.log",
                command,
                started_at=stage_started_at,
                ended_at=ended_at,
                returncode=None,
                stdout=exc.stdout,
                stderr=exc.stderr,
                timeout=True,
            )
            return _fail(
                args,
                arm=arm,
                failure_class="timeout",
                errors=[f"child timeout after {args.timeout_seconds} seconds"],
                returncode=None,
                prelaunch=prelaunch,
                completed=completed,
                payloads=payloads,
                started_at=started_at,
                stage_started_at=stage_started_at,
                start_memory=start_memory,
            )
        except OSError as exc:
            ended_at = _utc_now()
            _write_child_log(
                args.out / f"run-{index:02d}-{arm.stage}.log",
                command,
                started_at=stage_started_at,
                ended_at=ended_at,
                returncode=None,
                stdout="",
                stderr=str(exc),
            )
            return _fail(
                args,
                arm=arm,
                failure_class="launch_error",
                errors=[f"child launch error: {exc}"],
                returncode=None,
                prelaunch=prelaunch,
                completed=completed,
                payloads=payloads,
                started_at=started_at,
                stage_started_at=stage_started_at,
                start_memory=start_memory,
            )

        ended_at = _utc_now()
        _write_child_log(
            args.out / f"run-{index:02d}-{arm.stage}.log",
            command,
            started_at=stage_started_at,
            ended_at=ended_at,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if result.returncode != 0:
            failure_class = (
                "sigkill"
                if result.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL)
                else "child_failure"
            )
            return _fail(
                args,
                arm=arm,
                failure_class=failure_class,
                errors=[f"child returncode={result.returncode}"],
                returncode=result.returncode,
                prelaunch=prelaunch,
                completed=completed,
                payloads=payloads,
                started_at=started_at,
                stage_started_at=stage_started_at,
                start_memory=start_memory,
            )

        try:
            payload = _validate_artifact(
                metrics_path,
                arm,
                signature_before=signature_before,
                canonical_out=args.out,
            )
        except ArtifactError as exc:
            return _fail(
                args,
                arm=arm,
                failure_class=exc.failure_class,
                errors=exc.errors,
                returncode=result.returncode,
                prelaunch=prelaunch,
                completed=completed,
                payloads=payloads,
                failed_payload=(exc.payload if isinstance(exc.payload, dict) else None),
                started_at=started_at,
                stage_started_at=stage_started_at,
                start_memory=start_memory,
            )

        current_identity = _identity(payload)
        if reference_identity is None:
            reference_identity = current_identity
        else:
            mismatches = _identity_mismatches(reference_identity, current_identity, arm)
            if mismatches:
                return _fail(
                    args,
                    arm=arm,
                    failure_class="provenance_mismatch",
                    errors=mismatches,
                    returncode=result.returncode,
                    prelaunch=prelaunch,
                    completed=completed,
                    payloads=payloads,
                    failed_payload=payload,
                    started_at=started_at,
                    stage_started_at=stage_started_at,
                    start_memory=start_memory,
                )

        payloads[arm.stage] = payload
        completed.append(arm)
        _atomic_write_json(
            args.out / "supervisor-progress.json",
            _completed_supervisor_progress(
                status="stage_complete",
                started_at=started_at,
                arm=arm,
                completed=completed,
            ),
        )

    # The P0 child owns canonical metrics.json/results.md.  After validating the
    # complete three-arm result, the supervisor only updates its own progress
    # record; it must not rewrite or reformat the scientific artifacts.
    _atomic_write_json(
        args.out / "supervisor-progress.json",
        {
            **_completed_supervisor_progress(
                status="complete",
                started_at=started_at,
                arm=ARMS[-1],
                completed=completed,
            ),
            "protocol_complete": True,
            "ended_at": _utc_now(),
        },
    )
    _best_effort_write_json(
        args.out / "host-at-end.json",
        {
            "status": "complete",
            "protocol_complete": True,
            "memory": _proc_meminfo(),
        },
    )
    print(
        "supervised #85 protocol complete; canonical P0 artifacts validated", flush=True
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npy-root", type=Path, required=True)
    parser.add_argument("--npy-pattern", default="goz68-block_{block:03d}-attn")
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument(
        "--pack-pattern", default="block_{block:03d}-attention_plus_expert.goz1"
    )
    parser.add_argument("--embedding-shard", type=Path, required=True)
    parser.add_argument(
        "--int4-side-root",
        type=Path,
        required=True,
        help="Fresh external shared INT4/LS-alpha side-table cache root",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(DEFAULT_TIMEOUT_SECONDS),
        help="Explicit timeout applied independently to each arm child",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_SUPERVISOR_FAILCLOSED


if __name__ == "__main__":
    raise SystemExit(main())
