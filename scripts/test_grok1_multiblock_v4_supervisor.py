from __future__ import annotations

import json
import signal

# Test-only CompletedProcess/TimeoutExpired fixtures; no process is launched here.
import subprocess  # nosec B404
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grok1_multiblock_v4_supervisor as supervisor  # noqa: E402


_PACK_SHA = "a" * 64
_NPY_SHA = "b" * 64
_IMPLEMENTATION = {"commit": "c" * 40, "dirty": False}


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _spec_for_command(command: list[str]) -> supervisor.ArmSpec:
    arm = _value(command, "--arm")
    if arm == "int4":
        return supervisor.ARMS[0]
    if "--hp-blocks" in command:
        return supervisor.ARMS[1]
    return supervisor.ARMS[2]


def _payload(
    arm: supervisor.ArmSpec,
    *,
    pack_sha: str = _PACK_SHA,
    npy_sha: str = _NPY_SHA,
    implementation: dict | None = None,
    tokens: int = supervisor.TOKENS,
    decision: int = 2,
) -> dict:
    mode, hp_blocks, int4_blocks, channel_blocks = supervisor._schedule_for_arm(arm)
    sources = {
        block: (
            "fp16_control"
            if arm.stage == "p1" and block in (1, 2, 3)
            else (
                "research_int4_side"
                if arm.stage == "baseline"
                else "research_int4_channel_alpha_side"
            )
        )
        for block in supervisor.BLOCKS
    }
    cli_arm = "int4" if arm.stage == "baseline" else "int4_channel_alpha"
    provenance = {
        "issue": supervisor.ISSUE,
        "agent": supervisor.AGENT_LINE,
        "model": "Grok-4.5",
        "architecture_source": "github.com/xai-org/grok-1 model.py + run.py",
        "implementation": implementation or dict(_IMPLEMENTATION),
        "embedding_shard": "embedding.npy",
        "skip_fp16_control": False,
        "arm": cli_arm,
        "evidence_role": (
            "secondary; no independent decision"
            if arm.evidence_only
            else "primary; sole canonical #85 decision"
        ),
    }
    chain = {
        "arm_label": arm.expected_label,
        "expert_mode": mode,
        "tokens": tokens,
        "token_seed": supervisor.SEED,
        "blocks": list(supervisor.BLOCKS),
        "top_k": supervisor.TOP_K,
        "skip_fp16_control": False,
        "hp_blocks": hp_blocks,
        "int4_blocks": int4_blocks,
        "channel_alpha_blocks": channel_blocks,
        "per_block": [
            {
                "block": block,
                "expert_only": {
                    "block_output_cosine": 0.99,
                    "router_top1_agreement": 0.98,
                    "router_top2_set_agreement": 0.99,
                    "expert_load_js_bits": 0.01,
                    "block_output_drift_relative_norm": 0.02,
                    "residual_stream_in": {
                        "residual_in_drift_relative_norm": 0.01
                    },
                },
                "fp16_control": {"block_output_cosine": 1.0},
            }
            for block in supervisor.BLOCKS
        ],
        "end_of_chain": {
            "expert_only_chain_exit": {"residual_drift_relative_norm": 0.03}
        },
        "pack_provenance": [
            {
                "block": block,
                "pack_sha256": pack_sha,
                "npy_sha256": npy_sha,
                "scale_sources": {f"block_{block:03d}.expert": sources[block]},
            }
            for block in supervisor.BLOCKS
        ],
    }
    result = {"provenance": provenance, "chain": chain}
    if not arm.evidence_only:
        ranking = {
            "ordered_candidates": [
                "expert_int4_channel_alpha",
                "expert_int4_channel_alpha_123",
            ],
            "baseline_comparator": "expert_int4",
            "baseline_deltas": {
                "expert_int4_channel_alpha": {
                    "b3_top1_gain": 0.1,
                    "b3_cos_gain": 0.1,
                    "chain_exit_drift_reduction": 0.1,
                },
                "expert_int4_channel_alpha_123": {
                    "b3_top1_gain": 0.05,
                    "b3_cos_gain": 0.05,
                    "chain_exit_drift_reduction": 0.05,
                },
            },
            "winner": "expert_int4_channel_alpha",
            "tie_break_reason": "not needed",
        }
        result["comparison"] = {
            "protocol_complete": True,
            "completed_arms": [item.expected_label for item in supervisor.ARMS],
            "missing_arms": [],
            "invalid_arms": [],
            "validation_errors": [],
            "ordered_candidates": [
                "expert_int4_channel_alpha",
                "expert_int4_channel_alpha_123",
            ],
            "ranking": ranking,
            "best_remedy_arm": ranking["winner"],
        }
        result["decision"] = {
            "decision": decision,
            "decision_text": "fixture decision",
            "best_remedy_arm": "expert_int4_channel_alpha",
            "rationale": [],
            "protocol_complete": True,
            "ordered_candidates": ranking["ordered_candidates"],
            "baseline_deltas": ranking["baseline_deltas"],
            "tie_break_reason": ranking["tie_break_reason"],
        }
    return result


def _write_success(
    command: list[str], *, write_report: bool = True, **payload_kwargs: object
) -> str | None:
    spec = _spec_for_command(command)
    out = Path(_value(command, "--out"))
    out.mkdir(parents=True, exist_ok=True)
    metrics_body = json.dumps(_payload(spec, **payload_kwargs), indent=2) + "\n"
    (out / "metrics.json").write_text(metrics_body, encoding="utf-8")
    report_body = None
    if spec.stage == "p0" and write_report:
        report_body = "# Fixture\n\n**Option 2 — fixture decision**\n"
        supervisor._atomic_write_text(out / "results.md", report_body)
    progress = Path(_value(command, "--progress-json"))
    supervisor._atomic_write_json(
        progress,
        {
            "status": "running",
            "arm": _value(command, "--arm"),
            "protocol": {
                "tokens": supervisor.TOKENS,
                "seed": supervisor.SEED,
                "blocks": list(supervisor.BLOCKS),
                "top_k": supervisor.TOP_K,
            },
            "implementation": dict(_IMPLEMENTATION),
            "input_identity": {"embedding_shard": "embedding.npy"},
            "provenance_identity": {"model": "Grok-4.5"},
            "current_block": 3,
            "completed_blocks": list(supervisor.BLOCKS),
        },
    )
    return report_body


class SupervisorTests(unittest.TestCase):
    def _args(self, root: Path):
        return supervisor.build_parser().parse_args(
            [
                "--npy-root",
                str(root / "npy"),
                "--pack-root",
                str(root / "packs"),
                "--embedding-shard",
                str(root / "embedding.npy"),
                "--int4-side-root",
                str(root / "cache"),
                "--out",
                str(root / "report"),
                "--timeout-seconds",
                "123",
            ]
        )

    def _run(self, args, side_effect):
        memory = {
            "captured_at": "2026-08-23T00:00:00Z",
            "values": {"MemAvailable_bytes": 1},
            "launch_gate_applied": False,
        }
        with (
            mock.patch.object(
                supervisor.subprocess, "run", side_effect=side_effect
            ) as run,
            mock.patch.object(supervisor, "_proc_meminfo", return_value=memory),
        ):
            result = supervisor.run(args)
        return result, run

    def _failure(self, args) -> dict:
        return json.loads((args.out / "metrics.json").read_text(encoding="utf-8"))

    def test_locked_launch_order_arguments_and_successful_non_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)
            commands: list[list[str]] = []
            primary_bodies: dict[str, str] = {}

            def child(command, **kwargs):
                command = list(command)
                commands.append(command)
                spec = _spec_for_command(command)
                progress = json.loads(
                    Path(_value(command, "--progress-json")).read_text(encoding="utf-8")
                )
                self.assertEqual(progress["status"], "prelaunch")
                self.assertEqual(progress["stage"], spec.stage)
                if spec.stage == "p1":
                    self.assertTrue(
                        (args.out / "int4-baseline" / "metrics.json").is_file()
                    )
                if spec.stage == "p0":
                    self.assertTrue(
                        (args.out / "int4-channel-alpha-123" / "metrics.json").is_file()
                    )
                report = _write_success(command)
                if spec.stage == "p0":
                    primary_bodies["metrics"] = (args.out / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                    primary_bodies["report"] = report or ""
                self.assertEqual(kwargs["shell"], False)
                self.assertEqual(kwargs["capture_output"], True)
                self.assertEqual(kwargs["text"], True)
                self.assertEqual(kwargs["check"], False)
                self.assertEqual(kwargs["timeout"], 123.0)
                self.assertEqual(kwargs["cwd"], str(supervisor.REPO_ROOT))
                return subprocess.CompletedProcess(command, 0, "ok", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_OK)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(
                [_spec_for_command(cmd).stage for cmd in commands],
                ["baseline", "p1", "p0"],
            )
            for command in commands:
                self.assertEqual(command[0], sys.executable)
                self.assertEqual(command[1], str(supervisor.EXPERIMENT_SCRIPT))
                self.assertEqual(_value(command, "--tokens"), "8192")
                self.assertEqual(_value(command, "--seed"), "20260806")
                self.assertEqual(_value(command, "--blocks"), "0,1,2,3")
                self.assertEqual(_value(command, "--top-k"), "2")
                self.assertNotIn("--skip-fp16-control", command)
                self.assertNotIn("2048", command)
            self.assertEqual(
                Path(_value(commands[0], "--out")).name,
                "int4-baseline",
            )
            self.assertEqual(
                Path(_value(commands[1], "--out")).name,
                "int4-channel-alpha-123",
            )
            self.assertEqual(Path(_value(commands[2], "--out")), args.out)
            p0_comparisons = [
                commands[2][index + 1]
                for index, item in enumerate(commands[2])
                if item == "--comparison-metrics"
            ]
            self.assertEqual(
                [Path(path).parent.name for path in p0_comparisons],
                ["int4-baseline", "int4-channel-alpha-123"],
            )
            # The P0 child owns these. The supervisor validates but never rewrites them.
            self.assertEqual(
                (args.out / "metrics.json").read_text(encoding="utf-8"),
                primary_bodies["metrics"],
            )
            self.assertEqual(
                (args.out / "results.md").read_text(encoding="utf-8"),
                primary_bodies["report"],
            )
            final_progress = json.loads(
                (args.out / "supervisor-progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final_progress["status"], "complete")
            self.assertTrue(final_progress["protocol_complete"])
            self.assertEqual(
                json.loads((args.out / "host-at-launch.json").read_text())["status"],
                "prelaunch",
            )
            self.assertEqual(
                json.loads((args.out / "host-at-end.json").read_text())["status"],
                "complete",
            )

    def test_sigkill_minus_9_and_wrapper_137_fail_closed(self) -> None:
        for returncode in (-signal.SIGKILL, 137):
            with (
                self.subTest(returncode=returncode),
                tempfile.TemporaryDirectory() as td,
            ):
                args = self._args(Path(td))
                result, run = self._run(
                    args,
                    lambda command, return_code=returncode, **kwargs: (
                        subprocess.CompletedProcess(command, return_code, "", "killed")
                    ),
                )
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, 1)
                payload = self._failure(args)
                self.assertEqual(payload["decision"]["decision"], 4)
                self.assertFalse(payload["protocol_complete"])
                self.assertEqual(payload["failure_class"], "sigkill")
                self.assertEqual(payload["returncode"], returncode)
                self.assertEqual(payload["signal"], "SIGKILL")
                self.assertEqual(payload["completed_arms"], [])
                self.assertEqual(
                    payload["missing_arms"],
                    [
                        "expert_int4_channel_alpha_123",
                        "expert_int4_channel_alpha",
                    ],
                )
                self.assertEqual(payload["invalid_arms"], ["expert_int4"])
                self.assertEqual(
                    json.loads((args.out / "host-at-end.json").read_text())["status"],
                    "failed",
                )
                report = (args.out / "results.md").read_text(encoding="utf-8")
                self.assertIn("consistent with OOM or an external SIGKILL", report)
                self.assertIn("Option 4", report)

    def test_timeout_has_distinct_failure_class(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def timeout(command, **kwargs):
                raise subprocess.TimeoutExpired(
                    command, kwargs["timeout"], "partial", "late"
                )

            result, run = self._run(args, timeout)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "timeout")
            self.assertIsNone(payload["returncode"])
            self.assertIsNone(payload["signal"])
            log = (args.out / "run-01-baseline.log").read_text(encoding="utf-8")
            self.assertIn("timeout: true", log)
            self.assertIn("partial", log)

    def test_ordinary_nonzero_at_p0_preserves_prior_arm_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            calls = 0

            def child(command, **kwargs):
                nonlocal calls
                calls += 1
                if calls < 3:
                    _write_success(list(command))
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(command, 17, "", "failed")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 3)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "child_failure")
            self.assertEqual(payload["failed_arm"], "expert_int4_channel_alpha")
            self.assertEqual(payload["returncode"], 17)
            self.assertEqual(
                payload["completed_arms"],
                [
                    "expert_int4",
                    "expert_int4_channel_alpha_123",
                ],
            )
            self.assertEqual(payload["missing_arms"], [])
            self.assertEqual(payload["invalid_arms"], ["expert_int4_channel_alpha"])
            self.assertIn(
                "expert_int4",
                payload["provenance"]["completed_arm_provenance"],
            )

    def test_zero_exit_missing_malformed_or_stale_evidence_stops_sequence(
        self,
    ) -> None:
        cases = ("missing", "malformed", "stale")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                args = self._args(Path(td))
                if case == "stale":
                    stale = args.out / "int4-baseline" / "metrics.json"
                    stale.parent.mkdir(parents=True, exist_ok=True)
                    stale.write_text(
                        json.dumps(_payload(supervisor.ARMS[0])), encoding="utf-8"
                    )

                def child(command, current_case=case, **kwargs):
                    if current_case == "malformed":
                        out = Path(_value(list(command), "--out"))
                        out.mkdir(parents=True, exist_ok=True)
                        (out / "metrics.json").write_text("{broken", encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0, "", "")

                result, run = self._run(args, child)
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, 1)
                payload = self._failure(args)
                self.assertEqual(payload["failure_class"], "invalid_evidence")
                self.assertEqual(payload["failed_arm"], "expert_int4")

    def test_zero_exit_p0_metadata_touch_does_not_refresh_results_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "results.md").write_text(
                "# Stale fixture\n\n**Option 2 — obsolete winner and measurements**\n",
                encoding="utf-8",
            )

            def child(command, **kwargs):
                command = list(command)
                spec = _spec_for_command(command)
                _write_success(command, write_report=spec.stage != "p0")
                if spec.stage == "p0":
                    (args.out / "results.md").touch()
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 3)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "invalid_evidence")
            self.assertEqual(payload["failed_arm"], "expert_int4_channel_alpha")
            self.assertTrue(
                any(
                    "stale canonical results.md" in error
                    for error in payload["failure"]["errors"]
                )
            )

    def test_zero_exit_p0_atomic_report_replacement_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            args.out.mkdir(parents=True, exist_ok=True)
            original = "# Fixture\n\n**Option 2 — fixture decision**\n"
            (args.out / "results.md").write_text(original, encoding="utf-8")

            def child(command, **kwargs):
                _write_success(list(command))
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_OK)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(
                (args.out / "results.md").read_text(encoding="utf-8"),
                original,
            )

    def test_protocol_mismatch_is_distinct_and_never_retries_2048(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            commands: list[list[str]] = []

            def child(command, **kwargs):
                command = list(command)
                commands.append(command)
                _write_success(command, tokens=2048)
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "protocol_mismatch")
            self.assertEqual(payload["active_tokens"], 8192)
            self.assertFalse(any("2048" in command for command in commands))

    def test_overflowing_fp16_metric_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def child(command, **kwargs):
                command = list(command)
                spec = _spec_for_command(command)
                payload = _payload(spec)
                payload["chain"]["per_block"][0]["fp16_control"][
                    "block_output_cosine"
                ] = 10**1000
                out = Path(_value(command, "--out"))
                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "protocol_mismatch")
            self.assertEqual(payload["completed_arms"], [])
            self.assertEqual(payload["invalid_arms"], ["expert_int4"])

    def test_malformed_baseline_metrics_never_count_as_completed(self) -> None:
        cases = {
            "missing": lambda payload: payload["chain"]["per_block"][0][
                "expert_only"
            ].pop("router_top1_agreement"),
            "non_finite": lambda payload: payload["chain"]["per_block"][0][
                "expert_only"
            ].__setitem__("router_top1_agreement", float("nan")),
            "out_of_domain": lambda payload: payload["chain"]["per_block"][0][
                "expert_only"
            ].__setitem__("router_top2_set_agreement", 2.0),
            "missing_chain_exit": lambda payload: payload["chain"].pop(
                "end_of_chain"
            ),
        }
        for case, corrupt in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                args = self._args(Path(td))

                def child(command, corrupt_case=corrupt, **kwargs):
                    command = list(command)
                    spec = _spec_for_command(command)
                    payload = _payload(spec)
                    corrupt_case(payload)
                    out = Path(_value(command, "--out"))
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "metrics.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
                    return subprocess.CompletedProcess(command, 0, "", "")

                result, run = self._run(args, child)
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, 1)
                failure = self._failure(args)
                self.assertEqual(failure["failure_class"], "invalid_evidence")
                self.assertEqual(failure["completed_arms"], [])
                self.assertEqual(failure["invalid_arms"], ["expert_int4"])

    def test_malformed_p1_metrics_stop_after_valid_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def child(command, **kwargs):
                command = list(command)
                spec = _spec_for_command(command)
                payload = _payload(spec)
                if spec.stage == "p1":
                    payload["chain"]["per_block"][0]["expert_only"][
                        "block_output_drift_relative_norm"
                    ] = -0.1
                out = Path(_value(command, "--out"))
                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 2)
            failure = self._failure(args)
            self.assertEqual(failure["failure_class"], "invalid_evidence")
            self.assertEqual(failure["completed_arms"], ["expert_int4"])
            self.assertEqual(
                failure["invalid_arms"], ["expert_int4_channel_alpha_123"]
            )

    def test_cross_arm_provenance_mismatch_is_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            calls = 0

            def child(command, **kwargs):
                nonlocal calls
                calls += 1
                command = list(command)
                _write_success(
                    command, pack_sha=("d" * 64 if calls == 2 else _PACK_SHA)
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 2)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "provenance_mismatch")
            self.assertEqual(payload["completed_arms"], ["expert_int4"])
            self.assertEqual(payload["failed_arm"], "expert_int4_channel_alpha_123")
            self.assertEqual(payload["missing_arms"], ["expert_int4_channel_alpha"])
            self.assertEqual(
                payload["provenance"]["failed_arm_artifact_provenance"]["pack_sha256"][
                    "0"
                ],
                "d" * 64,
            )
            self.assertNotIn("ordered_candidates", payload["comparison"])

    def test_zero_exit_primary_option_4_is_invalid_and_supervisor_exits_nonzero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def child(command, **kwargs):
                command = list(command)
                spec = _spec_for_command(command)
                _write_success(command, decision=(4 if spec.stage == "p0" else 2))
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 3)
            payload = self._failure(args)
            self.assertEqual(payload["decision"]["decision"], 4)
            self.assertEqual(payload["failure_class"], "invalid_evidence")
            self.assertFalse(payload["protocol_complete"])

    def test_sigkill_recovers_last_atomic_child_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def child(command, **kwargs):
                command = list(command)
                progress = Path(_value(command, "--progress-json"))
                supervisor._atomic_write_json(
                    progress,
                    {
                        "status": "running",
                        "arm": "int4",
                        "protocol": {
                            "tokens": 8192,
                            "seed": 20260806,
                            "blocks": [0, 1, 2, 3],
                            "top_k": 2,
                        },
                        "implementation": dict(_IMPLEMENTATION),
                        "input_identity": {"embedding_shard": "embedding.npy"},
                        "provenance_identity": {"model": "Grok-4.5"},
                        "current_block": 2,
                        "completed_blocks": [0, 1],
                    },
                )
                return subprocess.CompletedProcess(command, -signal.SIGKILL, "", "")

            result, _run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            payload = self._failure(args)
            self.assertEqual(payload["current_block"], 2)
            self.assertEqual(payload["completed_blocks"], [0, 1])
            self.assertEqual(payload["highest_completed_block"], 1)
            self.assertEqual(payload["active_tokens"], 8192)
            self.assertEqual(
                payload["provenance"]["failed_arm_progress_provenance"][
                    "implementation"
                ],
                _IMPLEMENTATION,
            )

    def test_atomic_progress_interruption_keeps_previous_final(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            progress = root / "progress.json"
            original = '{"status":"stable"}\n'
            progress.write_text(original, encoding="utf-8")
            with mock.patch.object(
                supervisor.os, "replace", side_effect=OSError("stop")
            ):
                with self.assertRaisesRegex(OSError, "stop"):
                    supervisor._atomic_write_json(progress, {"status": "new"})
            self.assertEqual(progress.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.glob(".progress.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
