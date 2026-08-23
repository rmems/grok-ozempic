from __future__ import annotations

import hashlib
import json
import os
import signal

# Test-only CompletedProcess/TimeoutExpired fixtures; no process is launched here.
import subprocess  # nosec B404
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grok1_multiblock_v4_supervisor as supervisor  # noqa: E402


_PACK_SHA = "a" * 64
_NPY_SHA = "b" * 64
_IMPLEMENTATION = {"commit": "c" * 40, "dirty": False}
_EMBEDDING_BYTES = b"gh85-v4-supervisor-embedding-fixture\n"
_EMBEDDING_SHA = hashlib.sha256(_EMBEDDING_BYTES).hexdigest()
_ARM_FIXTURES = {
    "baseline": {
        "schedule": ("int4", [], [0, 1, 2, 3], []),
        "sources": {block: "research_int4_side" for block in (0, 1, 2, 3)},
    },
    "p1": {
        "schedule": ("int4_channel_alpha", [1, 2, 3], [0], [0]),
        "sources": {
            0: "research_int4_channel_alpha_side",
            1: "fp16_control",
            2: "fp16_control",
            3: "fp16_control",
        },
    },
    "p0": {
        "schedule": (
            "int4_channel_alpha",
            [],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ),
        "sources": {
            block: "research_int4_channel_alpha_side" for block in (0, 1, 2, 3)
        },
    },
}

# Literal scientific inputs from the canonical 8192-token run.  These are
# intentionally different by stage so the supervisor tests exercise P1's real
# canonical win and Option 2 instead of accepting a self-consistent fake P0 win.
_STAGE_METRICS = {
    "baseline": {
        "rows": [
            (0.998636599415437, 0.0, 1.0, 1.0, 0.0, 0.052476083867624736),
            (
                0.9975672646063307,
                0.052476083867624736,
                0.9822998046875,
                0.9385986328125,
                0.00002033074208447681,
                0.0698795479982047,
            ),
            (
                0.9886573865540497,
                0.0698795479982047,
                0.92626953125,
                0.8861083984375,
                0.00019497360882110354,
                0.1504987802639529,
            ),
            (
                0.9805662657439359,
                0.1504987802639529,
                0.82373046875,
                0.7210693359375,
                0.0003170502785525748,
                0.19772214640333557,
            ),
        ],
        "exit": 0.19772214640333557,
    },
    "p1": {
        "rows": [
            (0.9981363940675975, 0.0, 1.0, 1.0, 0.0, 0.061248773387512494),
            (
                0.9982853808940991,
                0.061248773387512494,
                0.9820556640625,
                0.932861328125,
                0.000034276330251193764,
                0.058674898865502985,
            ),
            (
                0.9939599269144929,
                0.058674898865502985,
                0.9351806640625,
                0.9066162109375,
                0.00020028407461690186,
                0.10976319959391136,
            ),
            (
                0.9900038440611516,
                0.10976319959391136,
                0.8873291015625,
                0.8260498046875,
                0.00007148238084593193,
                0.14199439171490524,
            ),
        ],
        "exit": 0.14199439171490524,
    },
    "p0": {
        "rows": [
            (0.9981363940675975, 0.0, 1.0, 1.0, 0.0, 0.061248773387512494),
            (
                0.9970708376561523,
                0.061248773387512494,
                0.9820556640625,
                0.932861328125,
                0.000034276330251193764,
                0.07653289247308387,
            ),
            (
                0.9856984885711603,
                0.07653289247308387,
                0.9110107421875,
                0.8681640625,
                0.00034164579906311755,
                0.16877270290143606,
            ),
            (
                0.9761822196087365,
                0.16877270290143606,
                0.8017578125,
                0.6864013671875,
                0.0007234054126472283,
                0.22110917292671636,
            ),
        ],
        "exit": 0.22110917292671636,
    },
}

_EXPECTED_DELTAS = {
    "expert_int4_channel_alpha_123": {
        "b3_top1_gain": 0.0635986328125,
        "b3_cos_gain": 0.0094375783172157,
        "chain_exit_drift_reduction": 0.05572775468843033,
    },
    "expert_int4_channel_alpha": {
        "b3_top1_gain": -0.02197265625,
        "b3_cos_gain": -0.004384046135199382,
        "chain_exit_drift_reduction": -0.02338702652338079,
    },
}
_EXPECTED_ORDER = [
    "expert_int4_channel_alpha_123",
    "expert_int4_channel_alpha",
]
_EXPECTED_TIE_BREAK = (
    "not needed; winner has the highest viability/top-1/cosine/exit-drift rank"
)


def _summary(stage: str) -> dict:
    fixture = _STAGE_METRICS[stage]
    label = next(arm.expected_label for arm in supervisor.ARMS if arm.stage == stage)
    rows = fixture["rows"]
    return {
        "arm_label": label,
        "block_output_cosine": [row[0] for row in rows],
        "residual_in_drift": [row[1] for row in rows],
        "router_top1": [row[2] for row in rows],
        "router_top2": [row[3] for row in rows],
        "expert_load_js_bits": [row[4] for row in rows],
        "chain_exit_residual_drift": fixture["exit"],
        "compounding": "superlinear_or_runaway",
        "viable": False,
    }


def _ranking() -> dict:
    return {
        "ordered_candidates": list(_EXPECTED_ORDER),
        "baseline_comparator": "expert_int4",
        "baseline_deltas": json.loads(json.dumps(_EXPECTED_DELTAS)),
        "winner": "expert_int4_channel_alpha_123",
        "tie_break_reason": _EXPECTED_TIE_BREAK,
    }


def _install_contract(payload: dict, ranking: dict, option: int) -> None:
    """Install a mutually consistent but not necessarily truthful contract."""
    ranking_copy = json.loads(json.dumps(ranking))
    comparison = payload["comparison"]
    comparison["ranking"] = ranking_copy
    comparison["ordered_candidates"] = list(ranking_copy["ordered_candidates"])
    comparison["best_remedy_arm"] = ranking_copy["winner"]
    decision = payload["decision"]
    decision["decision"] = option
    decision["best_remedy_arm"] = ranking_copy["winner"]
    decision["ordered_candidates"] = list(ranking_copy["ordered_candidates"])
    decision["baseline_deltas"] = json.loads(
        json.dumps(ranking_copy["baseline_deltas"])
    )
    decision["tie_break_reason"] = ranking_copy["tie_break_reason"]


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
    embedding_sha256: object = _EMBEDDING_SHA,
    tokens: int = supervisor.TOKENS,
    decision: int = 2,
) -> dict:
    fixture = _ARM_FIXTURES[arm.stage]
    mode, hp_blocks, int4_blocks, channel_blocks = fixture["schedule"]
    sources = fixture["sources"]
    cli_arm = "int4" if arm.stage == "baseline" else "int4_channel_alpha"
    provenance = {
        "issue": supervisor.ISSUE,
        "agent": supervisor.AGENT_LINE,
        "model": "Grok-4.5",
        "architecture_source": "github.com/xai-org/grok-1 model.py + run.py",
        "implementation": (
            dict(_IMPLEMENTATION) if implementation is None else implementation
        ),
        "embedding_shard": "embedding.npy",
        "embedding_sha256": embedding_sha256,
        "numpy": "2.5.1",
        "python": "3.14.6",
        "skip_fp16_control": False,
        "arm": cli_arm,
        "evidence_role": (
            "secondary; no independent decision"
            if arm.evidence_only
            else "primary; sole canonical #85 decision"
        ),
    }
    stage_metrics = _STAGE_METRICS[arm.stage]
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
                    "block_output_cosine": metrics[0],
                    "router_top1_agreement": metrics[2],
                    "router_top2_set_agreement": metrics[3],
                    "expert_load_js_bits": metrics[4],
                    "block_output_drift_relative_norm": metrics[5],
                    "residual_stream_in": {
                        "residual_in_drift_relative_norm": metrics[1]
                    },
                },
                "fp16_control": {"block_output_cosine": 1.0},
            }
            for block, metrics in zip(
                supervisor.BLOCKS, stage_metrics["rows"], strict=True
            )
        ],
        "end_of_chain": {
            "expert_only_chain_exit": {
                "residual_drift_relative_norm": stage_metrics["exit"]
            }
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
        ranking = _ranking()
        baseline_payload = _payload(
            supervisor.ARMS[0],
            pack_sha=pack_sha,
            npy_sha=npy_sha,
            implementation=implementation,
            embedding_sha256=embedding_sha256,
            tokens=tokens,
        )
        p1_payload = _payload(
            supervisor.ARMS[1],
            pack_sha=pack_sha,
            npy_sha=npy_sha,
            implementation=implementation,
            embedding_sha256=embedding_sha256,
            tokens=tokens,
        )
        result["comparison"] = {
            "protocol_complete": True,
            "completed_arms": [item.expected_label for item in supervisor.ARMS],
            "missing_arms": [],
            "invalid_arms": [],
            "validation_errors": [],
            "secondary_arms": {
                supervisor.ARMS[0].expected_label: baseline_payload,
                supervisor.ARMS[1].expected_label: p1_payload,
            },
            "summaries": {
                arm_spec.expected_label: _summary(arm_spec.stage)
                for arm_spec in supervisor.ARMS
            },
            "ordered_candidates": list(_EXPECTED_ORDER),
            "ranking": ranking,
            "best_remedy_arm": ranking["winner"],
        }
        result["decision"] = {
            "decision": decision,
            "decision_text": "fixture decision",
            "best_remedy_arm": "expert_int4_channel_alpha_123",
            "rationale": [],
            "protocol_complete": True,
            "ordered_candidates": list(ranking["ordered_candidates"]),
            "baseline_deltas": json.loads(json.dumps(ranking["baseline_deltas"])),
            "tie_break_reason": ranking["tie_break_reason"],
            "compounding": "superlinear_or_runaway",
        }
    return result


def _write_success(
    command: list[str],
    *,
    write_report: bool = True,
    mutate=None,
    **payload_kwargs: object,
) -> str | None:
    spec = _spec_for_command(command)
    out = Path(_value(command, "--out"))
    out.mkdir(parents=True, exist_ok=True)
    payload_kwargs.setdefault(
        "embedding_sha256",
        (
            _value(command, "--embedding-sha256")
            if "--embedding-sha256" in command
            else _EMBEDDING_SHA
        ),
    )
    payload = _payload(spec, **payload_kwargs)
    if mutate is not None:
        mutate(payload)
    metrics_body = json.dumps(payload, indent=2) + "\n"
    (out / "metrics.json").write_text(metrics_body, encoding="utf-8")
    report_body = None
    if spec.stage == "p0" and write_report:
        report_option = payload["decision"]["decision"]
        report_body = f"# Fixture\n\n**Option {report_option} — fixture decision**\n"
        supervisor._atomic_write_text(out / "results.md", report_body)
    progress = Path(_value(command, "--progress-json"))
    supervisor._atomic_write_json(
        progress,
        {
            "status": "running",
            "arm": _value(command, "--arm"),
            "arm_label": spec.expected_label,
            "hp_blocks": _ARM_FIXTURES[spec.stage]["schedule"][1],
            "protocol": {
                "tokens": supervisor.TOKENS,
                "seed": supervisor.SEED,
                "blocks": list(supervisor.BLOCKS),
                "top_k": supervisor.TOP_K,
            },
            "implementation": dict(_IMPLEMENTATION),
            "input_identity": {
                "embedding_shard": "embedding.npy",
                "embedding_sha256": payload_kwargs["embedding_sha256"],
            },
            "embedding_sha256": payload_kwargs["embedding_sha256"],
            "provenance_identity": {"model": "Grok-4.5"},
            "current_block": 3,
            "completed_blocks": list(supervisor.BLOCKS),
        },
    )
    return report_body


class SupervisorTests(unittest.TestCase):
    def _args(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        (root / "embedding.npy").write_bytes(_EMBEDDING_BYTES)
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

    def _assert_output_lock_held(self, out: Path) -> None:
        lock_path = supervisor._output_lock_path(out)
        fd = os.open(lock_path, os.O_RDWR)
        try:
            with self.assertRaises(BlockingIOError):
                supervisor._fcntl.flock(
                    fd,
                    supervisor._fcntl.LOCK_EX | supervisor._fcntl.LOCK_NB,
                )
        finally:
            os.close(fd)

    def test_output_lock_is_persistent_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report"
            with supervisor._supervisor_output_lock(out) as locked_out:
                self.assertEqual(locked_out, out)
                self._assert_output_lock_held(out)

            lock_path = supervisor._output_lock_path(out)
            self.assertTrue(lock_path.is_file())
            fd = os.open(lock_path, os.O_RDWR)
            try:
                supervisor._fcntl.flock(
                    fd,
                    supervisor._fcntl.LOCK_EX | supervisor._fcntl.LOCK_NB,
                )
            finally:
                os.close(fd)

    def test_run_holds_output_lock_through_failure_publication(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def fail_run(_args, _state):
                self._assert_output_lock_held(args.out)
                raise RuntimeError("synthetic transaction failure")

            def publish_failure(_args, _state, _exc):
                self._assert_output_lock_held(args.out)
                return supervisor.EXIT_SUPERVISOR_FAILCLOSED

            with (
                mock.patch.object(supervisor, "_run_supervised", side_effect=fail_run),
                mock.patch.object(
                    supervisor,
                    "_publish_supervisor_exception",
                    side_effect=publish_failure,
                ) as publish,
            ):
                result = supervisor.run(args)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            publish.assert_called_once()
            lock_path = supervisor._output_lock_path(args.out)
            fd = os.open(lock_path, os.O_RDWR)
            try:
                supervisor._fcntl.flock(
                    fd,
                    supervisor._fcntl.LOCK_EX | supervisor._fcntl.LOCK_NB,
                )
            finally:
                os.close(fd)

    def test_interrupted_lock_wait_does_not_mutate_active_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            args.out.mkdir(parents=True, exist_ok=True)
            metrics = args.out / "metrics.json"
            report = args.out / "results.md"
            metrics.write_text("active metrics\n", encoding="utf-8")
            report.write_text("active report\n", encoding="utf-8")

            with mock.patch.object(
                supervisor._fcntl,
                "flock",
                side_effect=supervisor.SupervisorSignal(signal.SIGTERM),
            ):
                result = supervisor.run(args)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(metrics.read_text(encoding="utf-8"), "active metrics\n")
            self.assertEqual(report.read_text(encoding="utf-8"), "active report\n")

    def test_arm_schedules_match_literal_evidence_fixtures(self) -> None:
        for arm in supervisor.ARMS:
            with self.subTest(stage=arm.stage):
                self.assertEqual(
                    supervisor._schedule_for_arm(arm),
                    _ARM_FIXTURES[arm.stage]["schedule"],
                )

    def test_literal_chain_summaries_rank_p1_and_derive_option_2(self) -> None:
        summaries = {
            arm.expected_label: supervisor._ranking_summary(_payload(arm)["chain"])
            for arm in supervisor.ARMS
        }
        for arm in supervisor.ARMS:
            self.assertEqual(
                summaries[arm.expected_label],
                _summary(arm.stage),
            )
        ranking = supervisor._expected_ranking(summaries)
        self.assertEqual(ranking, _ranking())
        self.assertEqual(
            supervisor._expected_decision_option(summaries, ranking),
            2,
        )

    def test_exact_candidate_rank_tie_prefers_lower_complexity_p0(self) -> None:
        summaries = {arm.expected_label: _summary(arm.stage) for arm in supervisor.ARMS}
        summaries["expert_int4_channel_alpha_123"] = json.loads(
            json.dumps(summaries["expert_int4_channel_alpha"])
        )
        summaries["expert_int4_channel_alpha_123"]["arm_label"] = (
            "expert_int4_channel_alpha_123"
        )
        ranking = supervisor._expected_ranking(summaries)
        self.assertEqual(
            ranking["ordered_candidates"],
            [
                "expert_int4_channel_alpha",
                "expert_int4_channel_alpha_123",
            ],
        )
        self.assertEqual(ranking["winner"], "expert_int4_channel_alpha")
        self.assertIn("exact metric-rank tie", ranking["tie_break_reason"])

    def test_independent_option_helper_covers_viable_and_no_improvement(self) -> None:
        summaries = {
            arm.expected_label: supervisor._ranking_summary(_payload(arm)["chain"])
            for arm in supervisor.ARMS
        }
        no_improvement = json.loads(json.dumps(summaries))
        baseline = no_improvement["expert_int4"]
        for label in (
            "expert_int4_channel_alpha",
            "expert_int4_channel_alpha_123",
        ):
            no_improvement[label] = json.loads(json.dumps(baseline))
            no_improvement[label]["arm_label"] = label
        ranking = supervisor._expected_ranking(no_improvement)
        self.assertEqual(
            supervisor._expected_decision_option(no_improvement, ranking), 3
        )

        viable = json.loads(json.dumps(summaries))
        p1 = viable["expert_int4_channel_alpha_123"]
        p1["block_output_cosine"] = [0.99] * 4
        p1["router_top1"] = [0.99] * 4
        p1["router_top2"] = [0.95] * 4
        p1["chain_exit_residual_drift"] = 0.1
        p1["viable"] = True
        ranking = supervisor._expected_ranking(viable)
        self.assertEqual(ranking["winner"], "expert_int4_channel_alpha_123")
        self.assertEqual(supervisor._expected_decision_option(viable, ranking), 1)

    def test_embedding_fingerprint_is_stable_and_replacement_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)
            fingerprint = supervisor._embedding_fingerprint(args.embedding_shard)
            self.assertEqual(fingerprint["sha256"], _EMBEDDING_SHA)
            self.assertEqual(
                supervisor._embedding_identity_errors(
                    args.embedding_shard, fingerprint
                ),
                [],
            )
            replacement = root / "replacement.npy"
            replacement.write_bytes(b"x" * len(_EMBEDDING_BYTES))
            replacement.replace(args.embedding_shard)
            self.assertTrue(
                supervisor._embedding_identity_errors(args.embedding_shard, fingerprint)
            )

    def test_embedding_fingerprint_rejects_change_during_streamed_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            real_fstat = supervisor.os.fstat
            calls = 0

            def changed_fstat(fd):
                nonlocal calls
                calls += 1
                observed = real_fstat(fd)
                if calls == 1:
                    return observed
                changed = mock.Mock()
                changed.st_dev = observed.st_dev
                changed.st_ino = observed.st_ino
                changed.st_size = observed.st_size
                changed.st_mtime_ns = observed.st_mtime_ns
                changed.st_ctime_ns = observed.st_ctime_ns + 1
                return changed

            with mock.patch.object(supervisor.os, "fstat", side_effect=changed_fstat):
                with self.assertRaisesRegex(OSError, "changed while hashing"):
                    supervisor._embedding_fingerprint(args.embedding_shard)

    def test_early_interrupts_replace_stale_success_with_current_option4(
        self,
    ) -> None:
        phases = (
            "out_expanduser",
            "out_mkdir",
            "npy_expanduser",
            "meminfo",
            "prelaunch_write",
        )
        signals = (
            ("sigterm", lambda: supervisor.SupervisorSignal(signal.SIGTERM), "SIGTERM"),
            ("keyboard", KeyboardInterrupt, "SIGINT"),
        )

        for phase in phases:
            for signal_name, make_exception, expected_signal in signals:
                with (
                    self.subTest(phase=phase, signal=signal_name),
                    tempfile.TemporaryDirectory() as td,
                ):
                    root = Path(td)
                    args = self._args(root)
                    args.out.mkdir(parents=True, exist_ok=True)
                    (args.out / "metrics.json").write_text(
                        json.dumps({"decision": {"decision": 2}}), encoding="utf-8"
                    )
                    (args.out / "results.md").write_text(
                        "# stale success\n\n**Option 2 — stale**\n", encoding="utf-8"
                    )
                    supervisor._arm_progress(args.out, supervisor.ARMS[0]).write_text(
                        json.dumps(
                            {
                                "current_block": 3,
                                "completed_blocks": [0, 1, 2, 3],
                                "input_identity": {"npy_root": "/stale/private/npy"},
                            }
                        ),
                        encoding="utf-8",
                    )
                    memory = {
                        "captured_at": "2026-08-23T00:00:00Z",
                        "values": {"MemAvailable_bytes": 1},
                        "launch_gate_applied": False,
                    }
                    tripped = False
                    exception = make_exception()
                    real_expanduser = Path.expanduser
                    real_mkdir = Path.mkdir
                    real_atomic_write_json = supervisor._atomic_write_json

                    def interrupt_expanduser(
                        path,
                        current_args=args,
                        current_exception=exception,
                        current_phase=phase,
                        real=real_expanduser,
                    ):
                        nonlocal tripped
                        target = (
                            current_args.npy_root
                            if current_phase == "npy_expanduser"
                            else current_args.out
                        )
                        if not tripped and path == target:
                            tripped = True
                            raise current_exception
                        return real(path)

                    def interrupt_mkdir(
                        path,
                        *call_args,
                        current_args=args,
                        current_exception=exception,
                        real=real_mkdir,
                        **call_kwargs,
                    ):
                        nonlocal tripped
                        if not tripped and path == current_args.out:
                            tripped = True
                            raise current_exception
                        return real(path, *call_args, **call_kwargs)

                    def interrupt_meminfo(
                        current_exception=exception,
                        current_memory=memory,
                    ):
                        nonlocal tripped
                        if not tripped:
                            tripped = True
                            raise current_exception
                        return current_memory

                    def interrupt_prelaunch_write(
                        path,
                        payload,
                        current_args=args,
                        current_exception=exception,
                        real=real_atomic_write_json,
                    ):
                        nonlocal tripped
                        if not tripped and path == supervisor._arm_progress(
                            current_args.out, supervisor.ARMS[0]
                        ):
                            tripped = True
                            raise current_exception
                        return real(path, payload)

                    phase_patch = {
                        "out_expanduser": mock.patch.object(
                            Path,
                            "expanduser",
                            autospec=True,
                            side_effect=interrupt_expanduser,
                        ),
                        "npy_expanduser": mock.patch.object(
                            Path,
                            "expanduser",
                            autospec=True,
                            side_effect=interrupt_expanduser,
                        ),
                        "out_mkdir": mock.patch.object(
                            Path,
                            "mkdir",
                            autospec=True,
                            side_effect=interrupt_mkdir,
                        ),
                        "meminfo": mock.patch.object(
                            supervisor,
                            "_proc_meminfo",
                            side_effect=interrupt_meminfo,
                        ),
                        "prelaunch_write": mock.patch.object(
                            supervisor,
                            "_atomic_write_json",
                            side_effect=interrupt_prelaunch_write,
                        ),
                    }[phase]
                    memory_patch = (
                        mock.patch.object(
                            supervisor, "_proc_meminfo", return_value=memory
                        )
                        if phase != "meminfo"
                        else nullcontext()
                    )

                    with (
                        phase_patch,
                        memory_patch,
                        mock.patch.object(supervisor.subprocess, "run") as child_run,
                    ):
                        result = supervisor.run(args)

                    self.assertTrue(tripped)
                    self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                    self.assertEqual(child_run.call_count, 0)
                    failure = self._failure(args)
                    self.assertEqual(failure["decision"]["decision"], 4)
                    self.assertFalse(failure["protocol_complete"])
                    self.assertEqual(failure["failure_class"], "interrupted")
                    self.assertEqual(failure["signal"], expected_signal)
                    self.assertIsNone(failure["current_block"])
                    self.assertEqual(failure["completed_blocks"], [])
                    serialized = json.dumps(failure)
                    self.assertNotIn("/stale/private", serialized)
                    self.assertNotIn("stale success", serialized)

    def test_bootstrap_interrupt_still_invalidates_stale_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "metrics.json").write_text(
                json.dumps({"decision": {"decision": 2}}), encoding="utf-8"
            )
            real_utc_now = supervisor._utc_now
            calls = 0

            def interrupt_first_timestamp():
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise supervisor.SupervisorSignal(signal.SIGTERM)
                return real_utc_now()

            with (
                mock.patch.object(
                    supervisor, "_utc_now", side_effect=interrupt_first_timestamp
                ),
                mock.patch.object(supervisor.subprocess, "run") as child_run,
            ):
                result = supervisor.run(args)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(child_run.call_count, 0)
            failure = self._failure(args)
            self.assertEqual(failure["failure_class"], "interrupted")
            self.assertEqual(failure["failed_arm"], "expert_int4")
            self.assertEqual(failure["completed_arms"], [])
            self.assertEqual(failure["completed_blocks"], [])

    def test_transition_interrupts_keep_prior_arms_and_fail_the_next_arm(
        self,
    ) -> None:
        cases = (
            ("p1", 1, ["expert_int4"]),
            (
                "p0",
                2,
                ["expert_int4", "expert_int4_channel_alpha_123"],
            ),
        )
        phases = ("prelaunch", "arm_progress", "supervisor_progress")
        memory = {
            "captured_at": "2026-08-23T00:00:00Z",
            "values": {"MemAvailable_bytes": 1},
            "launch_gate_applied": False,
        }

        for target_stage, expected_calls, expected_completed in cases:
            target = next(arm for arm in supervisor.ARMS if arm.stage == target_stage)
            for phase in phases:
                with (
                    self.subTest(target_stage=target_stage, phase=phase),
                    tempfile.TemporaryDirectory() as td,
                ):
                    args = self._args(Path(td))
                    tripped = False
                    real_prelaunch = supervisor._prelaunch_progress
                    real_atomic_write_json = supervisor._atomic_write_json

                    def child(command, **_kwargs):
                        _write_success(list(command))
                        return subprocess.CompletedProcess(command, 0, "", "")

                    def interrupt_prelaunch(
                        *call_args,
                        current_phase=phase,
                        current_stage=target_stage,
                        real=real_prelaunch,
                        **call_kwargs,
                    ):
                        nonlocal tripped
                        arm = call_args[1]
                        if (
                            not tripped
                            and current_phase == "prelaunch"
                            and arm.stage == current_stage
                        ):
                            tripped = True
                            raise supervisor.SupervisorSignal(signal.SIGTERM)
                        return real(*call_args, **call_kwargs)

                    def interrupt_progress_write(
                        path,
                        payload,
                        current_args=args,
                        current_phase=phase,
                        current_stage=target_stage,
                        current_target=target,
                        real=real_atomic_write_json,
                    ):
                        nonlocal tripped
                        is_target_prelaunch = (
                            isinstance(payload, dict)
                            and payload.get("status") == "prelaunch"
                            and payload.get("stage") == current_stage
                        )
                        should_interrupt = (
                            current_phase == "arm_progress"
                            and path
                            == supervisor._arm_progress(
                                current_args.out, current_target
                            )
                        ) or (
                            current_phase == "supervisor_progress"
                            and path == current_args.out / "supervisor-progress.json"
                        )
                        if not tripped and is_target_prelaunch and should_interrupt:
                            tripped = True
                            raise supervisor.SupervisorSignal(signal.SIGTERM)
                        return real(path, payload)

                    with (
                        mock.patch.object(
                            supervisor,
                            "_prelaunch_progress",
                            side_effect=interrupt_prelaunch,
                        ),
                        mock.patch.object(
                            supervisor,
                            "_atomic_write_json",
                            side_effect=interrupt_progress_write,
                        ),
                        mock.patch.object(
                            supervisor.subprocess, "run", side_effect=child
                        ) as child_run,
                        mock.patch.object(
                            supervisor, "_proc_meminfo", return_value=memory
                        ),
                    ):
                        result = supervisor.run(args)

                    self.assertTrue(tripped)
                    self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                    self.assertEqual(child_run.call_count, expected_calls)
                    failure = self._failure(args)
                    self.assertEqual(failure["failure_class"], "interrupted")
                    self.assertEqual(failure["failed_arm"], target.expected_label)
                    self.assertEqual(failure["completed_arms"], expected_completed)
                    self.assertEqual(failure["completed_blocks"], [])
                    self.assertEqual(
                        failure["progress"]["child"]["stage"], target_stage
                    )
                    supervisor_progress = json.loads(
                        (args.out / "supervisor-progress.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(
                        supervisor_progress["failed_arm"], target.expected_label
                    )
                    self.assertEqual(
                        supervisor_progress["completed_arms"], expected_completed
                    )

    def test_exception_path_invalidates_metrics_before_diagnostic_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            args.out.mkdir(parents=True, exist_ok=True)
            metrics = args.out / "metrics.json"
            report = args.out / "results.md"
            metrics.write_text("stale option 2\n", encoding="utf-8")
            report.write_text("stale report\n", encoding="utf-8")
            real_expanduser = Path.expanduser
            tripped = False

            def interrupt_first_expanduser(path):
                nonlocal tripped
                if not tripped and path == args.out:
                    tripped = True
                    raise RuntimeError("synthetic setup failure")
                return real_expanduser(path)

            with (
                mock.patch.object(
                    Path,
                    "expanduser",
                    autospec=True,
                    side_effect=interrupt_first_expanduser,
                ),
                mock.patch.object(
                    supervisor,
                    "_failure_payload",
                    side_effect=RuntimeError("synthetic diagnostic failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic diagnostic failure"),
            ):
                supervisor.run(args)

            self.assertTrue(tripped)
            self.assertFalse(metrics.exists())
            self.assertEqual(report.read_text(encoding="utf-8"), "stale report\n")

    def test_embedding_hash_interruption_uses_current_portable_prelaunch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            stale_progress = supervisor._arm_progress(args.out, supervisor.ARMS[0])
            stale_progress.parent.mkdir(parents=True, exist_ok=True)
            stale_progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "current_block": 3,
                        "completed_blocks": [0, 1, 2, 3],
                        "input_identity": {"npy_root": "/stale/private/npy"},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                supervisor,
                "_embedding_fingerprint",
                side_effect=MemoryError("synthetic hash interruption"),
            ):
                result, run = self._run(args, lambda *_args, **_kwargs: None)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 0)
            failure = self._failure(args)
            self.assertIsNone(failure["current_block"])
            self.assertEqual(failure["completed_blocks"], [])
            progress = failure["progress"]["child"]
            self.assertEqual(progress["supervisor_started_at"], failure["started_at"])
            self.assertEqual(progress["input_identity"]["npy_root"], "<NPY_ROOT>")
            self.assertEqual(
                progress["input_identity"]["embedding_shard"], "embedding.npy"
            )
            serialized = json.dumps(failure)
            self.assertNotIn("/stale/private", serialized)
            self.assertNotIn(str(args.npy_root), serialized)
            self.assertNotIn(str(args.pack_root), serialized)
            self.assertNotIn(str(args.int4_side_root), serialized)

    def test_missing_embedding_failure_redacts_every_known_host_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)
            args.embedding_shard.unlink()

            result, run = self._run(args, lambda *_args, **_kwargs: None)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 0)
            failure = self._failure(args)
            serialized = json.dumps(failure)
            self.assertIn("<EMBEDDING_SHARD>", serialized)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("/home/", serialized)
            self.assertNotIn("/tmp/", serialized)

    def test_relative_paths_do_not_corrupt_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            args.out = Path(".")
            args.npy_root = Path("npy")
            detail = "metrics.json: cosine=0.95 ratio=1/8192"
            recovered = {
                "metrics.json": detail,
                "measurements": ["drift=0.141994", "throughput=64 MiB/s"],
            }

            self.assertEqual(
                supervisor._portable_failure_detail(args, detail),
                detail,
            )
            self.assertEqual(
                supervisor._portable_failure_value(args, recovered),
                recovered,
            )

    def test_staging_cleanup_failure_uses_current_prelaunch_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            stale_progress = supervisor._arm_progress(args.out, supervisor.ARMS[0])
            stale_progress.parent.mkdir(parents=True, exist_ok=True)
            stale_progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "current_block": 3,
                        "completed_blocks": [0, 1, 2, 3],
                        "input_identity": {"npy_root": "/stale/private/npy"},
                    }
                ),
                encoding="utf-8",
            )
            staging = args.out / supervisor._P0_STAGING_NAME

            with mock.patch.object(
                supervisor,
                "_prepare_p0_staging",
                side_effect=OSError(f"could not prepare {staging}"),
            ):
                result, run = self._run(args, lambda *_args, **_kwargs: None)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 0)
            failure = self._failure(args)
            self.assertIsNone(failure["current_block"])
            self.assertEqual(failure["completed_blocks"], [])
            self.assertEqual(
                failure["progress"]["child"]["supervisor_started_at"],
                failure["started_at"],
            )
            serialized = json.dumps(failure)
            self.assertIn("<OUTPUT_ROOT>/.p0-staging", serialized)
            self.assertNotIn("/stale/private", serialized)
            self.assertNotIn(str(args.out), serialized)

    def test_second_staging_cleanup_failure_uses_current_p0_prelaunch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            stale_progress = supervisor._arm_progress(args.out, supervisor.ARMS[2])
            stale_progress.parent.mkdir(parents=True, exist_ok=True)
            stale_progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "current_block": 3,
                        "completed_blocks": [0, 1, 2, 3],
                        "input_identity": {"npy_root": "/stale/private/npy"},
                    }
                ),
                encoding="utf-8",
            )
            staging = args.out / supervisor._P0_STAGING_NAME

            def child(command, **_kwargs):
                command = list(command)
                _write_success(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(
                supervisor,
                "_prepare_p0_staging",
                side_effect=[None, OSError(f"could not prepare {staging}")],
            ):
                result, run = self._run(args, child)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 2)
            failure = self._failure(args)
            self.assertEqual(
                failure["completed_arms"],
                [
                    supervisor.ARMS[0].expected_label,
                    supervisor.ARMS[1].expected_label,
                ],
            )
            self.assertEqual(failure["failed_arm"], supervisor.ARMS[2].expected_label)
            self.assertIsNone(failure["current_block"])
            self.assertEqual(failure["completed_blocks"], [])
            progress = failure["progress"]["child"]
            self.assertEqual(progress["stage"], "p0")
            self.assertEqual(progress["input_identity"]["npy_root"], "<NPY_ROOT>")
            serialized = json.dumps(failure)
            self.assertIn("<OUTPUT_ROOT>/.p0-staging", serialized)
            self.assertNotIn("/stale/private", serialized)
            self.assertNotIn(str(args.out), serialized)

    def test_failure_payload_recursively_redacts_recovered_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def child(command, **_kwargs):
                progress_path = Path(_value(list(command), "--progress-json"))
                supervisor._atomic_write_json(
                    progress_path,
                    {
                        "status": "running",
                        "nested": {
                            str(args.npy_root): [
                                str(args.pack_root),
                                {"embedding": str(args.embedding_shard)},
                            ]
                        },
                    },
                )
                return subprocess.CompletedProcess(command, 1, "", "")

            result, run = self._run(args, child)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            failure = self._failure(args)
            recovered = failure["progress"]["child"]["nested"]
            self.assertEqual(
                recovered,
                {
                    "<NPY_ROOT>": [
                        "<PACK_ROOT>",
                        {"embedding": "<EMBEDDING_SHARD>"},
                    ]
                },
            )
            serialized = json.dumps(failure)
            self.assertNotIn(str(Path(td)), serialized)

    def test_embedding_digest_is_hashed_once_passed_and_echoed_by_all_arms(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            commands: list[list[str]] = []

            def child(command, **_kwargs):
                command = list(command)
                commands.append(command)
                self.assertEqual(_value(command, "--embedding-sha256"), _EMBEDDING_SHA)
                _write_success(command)
                payload = json.loads(
                    (Path(_value(command, "--out")) / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    payload["provenance"]["embedding_sha256"], _EMBEDDING_SHA
                )
                progress = json.loads(
                    Path(_value(command, "--progress-json")).read_text(encoding="utf-8")
                )
                self.assertEqual(progress["embedding_sha256"], _EMBEDDING_SHA)
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(
                supervisor,
                "_embedding_fingerprint",
                wraps=supervisor._embedding_fingerprint,
            ) as fingerprint:
                result, run = self._run(args, child)

            self.assertEqual(result, supervisor.EXIT_OK)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(fingerprint.call_count, 1)
            self.assertEqual(len(commands), 3)

    def test_dirty_or_unavailable_implementation_fails_before_baseline_acceptance(
        self,
    ) -> None:
        invalid = {
            "dirty": {"commit": "c" * 40, "dirty": True},
            "unknown_cleanliness": {"commit": "c" * 40, "dirty": None},
            "missing_commit": {"commit": None, "dirty": False},
            "short_commit": {"commit": "c" * 39, "dirty": False},
            "uppercase_commit": {"commit": "C" * 40, "dirty": False},
            "non_hex_commit": {"commit": "z" * 40, "dirty": False},
        }
        for case, implementation in invalid.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                args = self._args(Path(td))

                def child(command, current_implementation=implementation, **_kwargs):
                    _write_success(list(command), implementation=current_implementation)
                    return subprocess.CompletedProcess(command, 0, "", "")

                result, run = self._run(args, child)
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, 1)
                payload = self._failure(args)
                self.assertEqual(payload["failure_class"], "invalid_evidence")
                self.assertEqual(payload["completed_arms"], [])
                self.assertEqual(payload["invalid_arms"], ["expert_int4"])
                self.assertNotIn("ranking", payload["comparison"])

    def test_missing_or_blank_runtime_provenance_fails_before_acceptance(
        self,
    ) -> None:
        corruptions = {
            "missing_numpy": lambda payload: payload["provenance"].pop("numpy"),
            "blank_numpy": lambda payload: payload["provenance"].__setitem__(
                "numpy", " "
            ),
            "missing_python": lambda payload: payload["provenance"].pop("python"),
            "blank_python": lambda payload: payload["provenance"].__setitem__(
                "python", " "
            ),
        }
        for case, corrupt in corruptions.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                args = self._args(Path(td))

                def child(command, current_corrupt=corrupt, **_kwargs):
                    _write_success(list(command), mutate=current_corrupt)
                    return subprocess.CompletedProcess(command, 0, "", "")

                result, run = self._run(args, child)
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, 1)
                payload = self._failure(args)
                self.assertEqual(payload["failure_class"], "invalid_evidence")
                self.assertEqual(payload["completed_arms"], [])
                self.assertEqual(payload["invalid_arms"], ["expert_int4"])

    def test_missing_malformed_or_wrong_embedding_digest_fails_closed(self) -> None:
        corruptions = {
            "missing": lambda payload: payload["provenance"].pop("embedding_sha256"),
            "malformed": lambda payload: payload["provenance"].__setitem__(
                "embedding_sha256", "not-a-sha"
            ),
            "wrong": lambda payload: payload["provenance"].__setitem__(
                "embedding_sha256", "d" * 64
            ),
        }
        for case, corrupt in corruptions.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                args = self._args(Path(td))

                def child(command, current_corrupt=corrupt, **_kwargs):
                    _write_success(list(command), mutate=current_corrupt)
                    return subprocess.CompletedProcess(command, 0, "", "")

                result, run = self._run(args, child)
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, 1)
                payload = self._failure(args)
                self.assertEqual(payload["failure_class"], "invalid_evidence")
                self.assertEqual(payload["completed_arms"], [])
                self.assertNotIn("ranking", payload["comparison"])

    def test_embedding_replacement_during_arm_fails_at_the_guard_boundary(self) -> None:
        for replaced_stage, expected_calls, expected_completed in (
            ("baseline", 1, []),
            ("p1", 2, ["expert_int4"]),
        ):
            with (
                self.subTest(stage=replaced_stage),
                tempfile.TemporaryDirectory() as td,
            ):
                root = Path(td)
                args = self._args(root)

                def child(
                    command,
                    current_stage=replaced_stage,
                    current_root=root,
                    current_args=args,
                    **_kwargs,
                ):
                    command = list(command)
                    spec = _spec_for_command(command)
                    _write_success(command)
                    if spec.stage == current_stage:
                        replacement = current_root / f"replacement-{spec.stage}.npy"
                        replacement.write_bytes(b"x" * len(_EMBEDDING_BYTES))
                        replacement.replace(current_args.embedding_shard)
                    return subprocess.CompletedProcess(command, 0, "", "")

                result, run = self._run(args, child)
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, expected_calls)
                payload = self._failure(args)
                self.assertEqual(payload["failure_class"], "provenance_mismatch")
                self.assertEqual(payload["completed_arms"], expected_completed)
                self.assertNotIn("ranking", payload["comparison"])

    def test_locked_launch_order_arguments_and_supervised_p0_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)
            args.out.mkdir(parents=True, exist_ok=True)
            prior_metrics = "previous accepted metrics\n"
            prior_report = "previous accepted report\n"
            (args.out / "metrics.json").write_text(prior_metrics, encoding="utf-8")
            (args.out / "results.md").write_text(prior_report, encoding="utf-8")
            staging = args.out / supervisor._P0_STAGING_NAME
            staging.mkdir()
            (staging / "metrics.json").write_text(
                "stale staged metrics\n", encoding="utf-8"
            )
            (staging / "results.md").write_text(
                "stale staged report\n", encoding="utf-8"
            )
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
                    self.assertFalse((staging / "metrics.json").exists())
                    self.assertFalse((staging / "results.md").exists())
                    self.assertFalse((args.out / "metrics.json").exists())
                    self.assertEqual(
                        (args.out / "results.md").read_text(encoding="utf-8"),
                        prior_report,
                    )
                report = _write_success(command)
                if spec.stage == "p0":
                    staged_out = Path(_value(command, "--out"))
                    primary_bodies["metrics"] = (staged_out / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                    primary_bodies["report"] = report or ""
                    self.assertFalse((args.out / "metrics.json").exists())
                self.assertEqual(kwargs["shell"], False)
                self.assertEqual(kwargs["capture_output"], True)
                self.assertEqual(kwargs["text"], True)
                self.assertEqual(kwargs["check"], False)
                self.assertEqual(kwargs["timeout"], 123.0)
                self.assertEqual(kwargs["cwd"], str(supervisor.REPO_ROOT))
                return subprocess.CompletedProcess(command, 0, "ok", "")

            real_validate = supervisor._validate_artifact

            def validate_staged(*call_args, **call_kwargs):
                arm = call_args[1]
                if arm.stage == "p0":
                    self.assertEqual(call_args[0].parent, staging)
                    self.assertFalse((args.out / "metrics.json").exists())
                    self.assertEqual(
                        (args.out / "results.md").read_text(encoding="utf-8"),
                        prior_report,
                    )
                return real_validate(*call_args, **call_kwargs)

            with mock.patch.object(
                supervisor, "_validate_artifact", side_effect=validate_staged
            ):
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
            self.assertEqual(
                Path(_value(commands[2], "--out")),
                args.out / supervisor._P0_STAGING_NAME,
            )
            p0_comparisons = [
                commands[2][index + 1]
                for index, item in enumerate(commands[2])
                if item == "--comparison-metrics"
            ]
            self.assertEqual(
                [Path(path).parent.name for path in p0_comparisons],
                ["int4-baseline", "int4-channel-alpha-123"],
            )
            # Only the supervisor promotes the validated staged pair.
            self.assertEqual(
                json.loads((args.out / "metrics.json").read_text(encoding="utf-8")),
                json.loads(primary_bodies["metrics"]),
            )
            self.assertEqual(
                (args.out / "results.md").read_text(encoding="utf-8"),
                primary_bodies["report"],
            )
            self.assertFalse((staging / "metrics.json").exists())
            self.assertFalse((staging / "results.md").exists())
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

    def test_post_validation_bookkeeping_error_preserves_canonical_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            canonical: dict[str, str] = {}

            def child(command, **_kwargs):
                command = list(command)
                spec = _spec_for_command(command)
                report = _write_success(command)
                if spec.stage == "p0":
                    staged_out = Path(_value(command, "--out"))
                    canonical["metrics"] = (staged_out / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                    canonical["report"] = report or ""
                return subprocess.CompletedProcess(command, 0, "", "")

            real_atomic_write_json = supervisor._atomic_write_json

            def fail_final_progress(path: Path, payload: object) -> None:
                if (
                    path.name == "supervisor-progress.json"
                    and isinstance(payload, dict)
                    and payload.get("status") == "complete"
                ):
                    raise RuntimeError("final bookkeeping failed")
                real_atomic_write_json(path, payload)

            with mock.patch.object(
                supervisor,
                "_atomic_write_json",
                side_effect=fail_final_progress,
            ):
                result, run = self._run(args, child)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(
                json.loads((args.out / "metrics.json").read_text(encoding="utf-8")),
                json.loads(canonical["metrics"]),
            )
            self.assertEqual(
                (args.out / "results.md").read_text(encoding="utf-8"),
                canonical["report"],
            )
            metrics = self._failure(args)
            self.assertEqual(metrics["decision"]["decision"], 2)
            self.assertNotIn("failure_class", metrics)
            diagnostic = json.loads(
                (args.out / "supervisor-bookkeeping-error.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(diagnostic["status"], "bookkeeping_failed")
            self.assertTrue(diagnostic["canonical_protocol_validated"])
            self.assertEqual(
                diagnostic["canonical_artifacts_preserved"],
                ["metrics.json", "results.md"],
            )

    def test_failure_publication_stops_after_failed_metrics_invalidation_fsync(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            metrics_path = out / "metrics.json"
            report_path = out / "results.md"
            metrics_path.write_text("stale option 2\n", encoding="utf-8")
            report_path.write_text("stale report\n", encoding="utf-8")

            with (
                mock.patch.object(
                    supervisor,
                    "_fsync_directory_strict",
                    side_effect=OSError("synthetic directory fsync failure"),
                ) as fsync_directory,
            ):
                for attempt in ("initial", "supervisor-exception retry"):
                    with (
                        self.subTest(attempt=attempt),
                        self.assertRaisesRegex(
                            OSError, "synthetic directory fsync failure"
                        ),
                    ):
                        supervisor._publish_failure(out, {})

            self.assertEqual(fsync_directory.call_args_list, [mock.call(out)] * 2)
            self.assertFalse(metrics_path.exists())
            self.assertEqual(report_path.read_text(encoding="utf-8"), "stale report\n")

    def test_validated_success_publishes_metrics_last(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            events: list[tuple[str, Path, object | None]] = []
            payload = {"decision": {"decision": 2}}

            with (
                mock.patch.object(
                    supervisor,
                    "_durably_unlink",
                    side_effect=lambda path: events.append(("unlink", path, None)),
                ),
                mock.patch.object(
                    supervisor,
                    "_atomic_write_text",
                    side_effect=lambda path, body: events.append(
                        ("report", path, body)
                    ),
                ),
                mock.patch.object(
                    supervisor,
                    "_atomic_write_json",
                    side_effect=lambda path, body: events.append(
                        ("metrics", path, body)
                    ),
                ),
            ):
                supervisor._publish_validated_success(out, payload, "validated\n")

            self.assertEqual(
                events,
                [
                    ("unlink", out / "metrics.json", None),
                    ("report", out / "results.md", "validated\n"),
                    ("metrics", out / "metrics.json", payload),
                ],
            )

    def test_failed_success_promotion_never_marks_canonical_validated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def child(command, **_kwargs):
                _write_success(list(command))
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(
                    supervisor,
                    "_publish_validated_success",
                    side_effect=OSError("synthetic promotion failure"),
                ),
                mock.patch.object(
                    supervisor,
                    "_record_post_validation_exception",
                    wraps=supervisor._record_post_validation_exception,
                ) as post_validation,
            ):
                result, run = self._run(args, child)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 3)
            self.assertFalse(post_validation.called)
            failure = self._failure(args)
            self.assertEqual(failure["decision"]["decision"], 4)
            self.assertEqual(failure["failure_class"], "supervisor_error")
            self.assertIn(
                "synthetic promotion failure", " ".join(failure["failure"]["errors"])
            )

    def test_p0_container_memory_error_preserves_validated_canonical_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            canonical: dict[str, str] = {}

            def child(command, **_kwargs):
                command = list(command)
                spec = _spec_for_command(command)
                report = _write_success(command)
                if spec.stage == "p0":
                    staged_out = Path(_value(command, "--out"))
                    canonical["metrics"] = (staged_out / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                    canonical["report"] = report or ""
                return subprocess.CompletedProcess(command, 0, "", "")

            real_record = supervisor._record_validated_arm

            def fail_p0_record(payloads, completed, arm, payload) -> None:
                if arm.stage == "p0":
                    raise MemoryError("synthetic post-validation container failure")
                real_record(payloads, completed, arm, payload)

            with mock.patch.object(
                supervisor,
                "_record_validated_arm",
                side_effect=fail_p0_record,
            ):
                result, run = self._run(args, child)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(
                json.loads((args.out / "metrics.json").read_text(encoding="utf-8")),
                json.loads(canonical["metrics"]),
            )
            self.assertEqual(
                (args.out / "results.md").read_text(encoding="utf-8"),
                canonical["report"],
            )
            diagnostic = json.loads(
                (args.out / "supervisor-bookkeeping-error.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(diagnostic["canonical_protocol_validated"])
            self.assertIn("MemoryError", diagnostic["error"])

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
                    json.loads(
                        (args.out / "host-at-end.json").read_text(encoding="utf-8")
                    )["status"],
                    "failed",
                )
                report = (args.out / "results.md").read_text(encoding="utf-8")
                self.assertIn("consistent with OOM or an external SIGKILL", report)
                self.assertIn("Option 4", report)

    def test_p0_partial_staging_sigkill_never_exposes_option_1_to_3(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            args.out.mkdir(parents=True, exist_ok=True)
            prior_metrics = "previous accepted canonical metrics\n"
            (args.out / "metrics.json").write_text(prior_metrics, encoding="utf-8")

            def child(command, **_kwargs):
                command = list(command)
                spec = _spec_for_command(command)
                if spec.stage != "p0":
                    _write_success(command)
                    return subprocess.CompletedProcess(command, 0, "", "")
                self.assertFalse((args.out / "metrics.json").exists())
                _write_success(command, write_report=False)
                self.assertFalse((args.out / "metrics.json").exists())
                staged = Path(_value(command, "--out")) / "metrics.json"
                self.assertEqual(
                    json.loads(staged.read_text(encoding="utf-8"))["decision"][
                        "decision"
                    ],
                    2,
                )
                return subprocess.CompletedProcess(command, -signal.SIGKILL, "", "")

            result, run = self._run(args, child)

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 3)
            failure = self._failure(args)
            self.assertEqual(failure["decision"]["decision"], 4)
            self.assertEqual(failure["failure_class"], "sigkill")
            self.assertEqual(failure["failed_arm"], "expert_int4_channel_alpha")
            staging = args.out / supervisor._P0_STAGING_NAME
            self.assertFalse((staging / "metrics.json").exists())
            self.assertFalse((staging / "results.md").exists())

    def test_startup_clears_stale_p0_staging_before_early_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))
            staging = args.out / supervisor._P0_STAGING_NAME
            staging.mkdir(parents=True)
            for name in ("metrics.json", "results.md"):
                (staging / name).write_text(
                    "rejected stale evidence\n", encoding="utf-8"
                )

            result, run = self._run(
                args,
                lambda command, **_kwargs: subprocess.CompletedProcess(
                    command, -signal.SIGKILL, "", ""
                ),
            )

            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(self._failure(args)["decision"]["decision"], 4)
            self.assertFalse((staging / "metrics.json").exists())
            self.assertFalse((staging / "results.md").exists())

    def test_timeout_has_distinct_failure_class(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)

            def timeout(command, **kwargs):
                command = list(command)
                child_out = Path(_value(command, "--out"))
                raise subprocess.TimeoutExpired(
                    command,
                    kwargs["timeout"],
                    output=(
                        f"wrote {child_out / 'metrics.json'}\nratio=1/8192\n"
                    ).encode(),
                    stderr=(
                        f"failed {args.npy_root / 'block.npy'} "
                        f"from {args.pack_root / 'pack.goz1'}\n"
                    ).encode(),
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
            self.assertIn("wrote <HOST_PATH>/int4-baseline/metrics.json", log)
            self.assertIn("failed <NPY_ROOT>/block.npy", log)
            self.assertIn("from <PACK_ROOT>/pack.goz1", log)
            self.assertIn("ratio=1/8192", log)
            self.assertNotIn(str(root), log)
            self.assertNotIn("/home/", log)
            self.assertNotIn("/tmp/", log)

    def test_child_log_command_uses_portable_paths_without_mutating_argv(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root)
            command = supervisor._child_command(
                args,
                supervisor.ARMS[-1],
                embedding_sha256=_EMBEDDING_SHA,
            )
            original_command = list(command)
            log_path = root / "child.log"

            supervisor._write_child_log(
                log_path,
                command,
                started_at="2026-08-23T00:00:00Z",
                ended_at="2026-08-23T00:01:00Z",
                returncode=0,
                stdout=(
                    f"wrote {Path(_value(command, '--out')) / 'metrics.json'}\n"
                    "cos=0.990004 drift=1.4199439171490524e-01 "
                    "ratio=1/8192 throughput=64 MiB/s"
                ),
                stderr=(
                    f"npy={args.npy_root / 'block.npy'}\n"
                    f"pack={args.pack_root / 'pack.goz1'}\n"
                    f"embedding={args.embedding_shard}\n"
                    f"cache={args.int4_side_root / 'codes.npy'}\n"
                    f"script={supervisor.EXPERIMENT_SCRIPT}\n"
                    f"python={sys.executable}"
                ),
            )

            self.assertEqual(command, original_command)
            log = log_path.read_text(encoding="utf-8")
            command_line = next(
                line for line in log.splitlines() if line.startswith("command: ")
            )
            self.assertIn(
                "python3 scripts/grok1_multiblock_experiment.py", command_line
            )
            for placeholder in (
                "<NPY_ROOT>",
                "<PACK_ROOT>",
                "<EMBEDDING_SHARD>",
                "<INT4_SIDE_ROOT>",
                "<HOST_PATH>/.p0-staging",
            ):
                self.assertIn(placeholder, command_line)
            self.assertNotIn(str(root), command_line)
            self.assertNotIn(str(supervisor.REPO_ROOT), command_line)
            self.assertNotIn(sys.executable, command_line)
            self.assertNotIn(str(root), log)
            self.assertNotIn(str(supervisor.REPO_ROOT), log)
            self.assertNotIn(sys.executable, log)
            self.assertIn("wrote <HOST_PATH>/.p0-staging/metrics.json", log)
            self.assertIn("npy=<NPY_ROOT>/block.npy", log)
            self.assertIn("pack=<PACK_ROOT>/pack.goz1", log)
            self.assertIn("embedding=<EMBEDDING_SHARD>", log)
            self.assertIn("cache=<INT4_SIDE_ROOT>/codes.npy", log)
            self.assertIn("script=scripts/grok1_multiblock_experiment.py", log)
            self.assertIn("python=python3", log)
            self.assertIn(
                "cos=0.990004 drift=1.4199439171490524e-01 "
                "ratio=1/8192 throughput=64 MiB/s",
                log,
            )

    def test_launch_error_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def launch_error(_command, **_kwargs):
                raise OSError("exec format error")

            result, run = self._run(args, launch_error)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "launch_error")
            self.assertEqual(payload["decision"]["decision"], 4)
            self.assertIsNone(payload["returncode"])
            self.assertIn(
                "exec format error",
                (args.out / "run-01-baseline.log").read_text(encoding="utf-8"),
            )

    def test_keyboard_interrupt_publishes_option_4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def interrupted(_command, **_kwargs):
                raise KeyboardInterrupt

            result, run = self._run(args, interrupted)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "interrupted")
            self.assertEqual(payload["decision"]["decision"], 4)
            self.assertEqual(payload["returncode"], -signal.SIGINT)
            self.assertEqual(payload["signal"], "SIGINT")

    def test_unexpected_supervisor_error_publishes_option_4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def unexpected(_command, **_kwargs):
                raise RuntimeError("synthetic supervisor fault")

            result, run = self._run(args, unexpected)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "supervisor_error")
            self.assertEqual(payload["decision"]["decision"], 4)
            self.assertIsNone(payload["returncode"])
            self.assertIn(
                "RuntimeError: synthetic supervisor fault",
                payload["failure"]["errors"],
            )

    def test_sigterm_interrupt_publishes_option_4(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            args = self._args(Path(td))

            def terminated(_command, **_kwargs):
                supervisor._raise_supervisor_signal(signal.SIGTERM, None)

            result, run = self._run(args, terminated)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 1)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "interrupted")
            self.assertEqual(payload["decision"]["decision"], 4)
            self.assertEqual(payload["returncode"], -signal.SIGTERM)
            self.assertEqual(payload["signal"], "SIGTERM")

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

    def test_zero_exit_p0_missing_staged_report_fails_closed(self) -> None:
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
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 3)
            payload = self._failure(args)
            self.assertEqual(payload["failure_class"], "invalid_evidence")
            self.assertEqual(payload["failed_arm"], "expert_int4_channel_alpha")
            self.assertTrue(
                any(
                    "P0 results.md is missing" in error
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
                (out / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
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
            "missing_chain_exit": lambda payload: payload["chain"].pop("end_of_chain"),
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
                (out / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            result, run = self._run(args, child)
            self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
            self.assertEqual(run.call_count, 2)
            failure = self._failure(args)
            self.assertEqual(failure["failure_class"], "invalid_evidence")
            self.assertEqual(failure["completed_arms"], ["expert_int4"])
            self.assertEqual(failure["invalid_arms"], ["expert_int4_channel_alpha_123"])

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

    def test_cross_arm_runtime_version_mismatch_is_distinct(self) -> None:
        for field in ("numpy", "python"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                args = self._args(Path(td))
                calls = 0

                def child(command, current_field=field, **_kwargs):
                    nonlocal calls
                    calls += 1
                    command = list(command)

                    def change_runtime(current_payload, runtime_field=current_field):
                        current_payload["provenance"][runtime_field] = (
                            "different-version"
                        )

                    _write_success(
                        command,
                        mutate=change_runtime if calls == 2 else None,
                    )
                    return subprocess.CompletedProcess(command, 0, "", "")

                result, run = self._run(args, child)
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, 2)
                payload = self._failure(args)
                self.assertEqual(payload["failure_class"], "provenance_mismatch")
                self.assertEqual(payload["completed_arms"], ["expert_int4"])
                self.assertIn(
                    f"expert_int4_channel_alpha_123:{field}_mismatch",
                    payload["failure"]["errors"],
                )

    def test_self_consistent_forged_ranking_decision_and_summaries_fail_closed(
        self,
    ) -> None:
        def forged_winner(payload: dict) -> None:
            ranking = _ranking()
            ranking["ordered_candidates"] = list(reversed(_EXPECTED_ORDER))
            ranking["winner"] = "expert_int4_channel_alpha"
            _install_contract(payload, ranking, 2)

        def forged_delta(payload: dict) -> None:
            ranking = _ranking()
            ranking["baseline_deltas"]["expert_int4_channel_alpha_123"][
                "b3_top1_gain"
            ] += 0.25
            _install_contract(payload, ranking, 2)

        def forged_tie_reason(payload: dict) -> None:
            ranking = _ranking()
            ranking["tie_break_reason"] = (
                "exact metric-rank tie; preferred P0 despite unequal raw metrics"
            )
            _install_contract(payload, ranking, 2)

        def forged_option(payload: dict) -> None:
            _install_contract(payload, _ranking(), 3)

        def forged_nested_summary(payload: dict) -> None:
            comparison = payload["comparison"]
            p1 = comparison["summaries"]["expert_int4_channel_alpha_123"]
            p1["router_top1"][-1] = 0.5
            ranking = _ranking()
            ranking["ordered_candidates"] = list(reversed(_EXPECTED_ORDER))
            ranking["winner"] = "expert_int4_channel_alpha"
            ranking["baseline_deltas"]["expert_int4_channel_alpha_123"][
                "b3_top1_gain"
            ] = 0.5 - 0.82373046875
            _install_contract(payload, ranking, 3)

        def forged_nested_exact_tie(payload: dict) -> None:
            comparison = payload["comparison"]
            p0 = json.loads(
                json.dumps(comparison["summaries"]["expert_int4_channel_alpha"])
            )
            p0["arm_label"] = "expert_int4_channel_alpha_123"
            comparison["summaries"]["expert_int4_channel_alpha_123"] = p0
            ranking = _ranking()
            ranking["ordered_candidates"] = list(reversed(_EXPECTED_ORDER))
            ranking["winner"] = "expert_int4_channel_alpha"
            ranking["baseline_deltas"]["expert_int4_channel_alpha_123"] = json.loads(
                json.dumps(ranking["baseline_deltas"]["expert_int4_channel_alpha"])
            )
            ranking["tie_break_reason"] = (
                "exact metric-rank tie; preferred P0 "
                "expert_int4_channel_alpha as the lower-complexity remedy"
            )
            _install_contract(payload, ranking, 3)

        def forged_residual_summary(payload: dict) -> None:
            payload["comparison"]["summaries"]["expert_int4_channel_alpha_123"][
                "residual_in_drift"
            ][-1] = 0.001

        def forged_js_summary(payload: dict) -> None:
            payload["comparison"]["summaries"]["expert_int4_channel_alpha_123"][
                "expert_load_js_bits"
            ][-1] = 0.5

        def forged_compounding(payload: dict) -> None:
            payload["comparison"]["summaries"]["expert_int4_channel_alpha_123"][
                "compounding"
            ] = "sublinear_or_saturating"
            payload["decision"]["compounding"] = "sublinear_or_saturating"

        def forged_decision_compounding(payload: dict) -> None:
            payload["decision"]["compounding"] = "roughly_linear"

        cases = {
            "winner_and_order": forged_winner,
            "baseline_delta": forged_delta,
            "tie_break": forged_tie_reason,
            "decision_option": forged_option,
            "nested_summary": forged_nested_summary,
            "nested_exact_tie": forged_nested_exact_tie,
            "residual_summary": forged_residual_summary,
            "js_summary": forged_js_summary,
            "compounding_summary": forged_compounding,
            "decision_compounding": forged_decision_compounding,
        }
        for case, mutate in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                args = self._args(Path(td))

                def child(command, current_mutate=mutate, **_kwargs):
                    command = list(command)
                    spec = _spec_for_command(command)
                    _write_success(
                        command,
                        mutate=(current_mutate if spec.stage == "p0" else None),
                    )
                    return subprocess.CompletedProcess(command, 0, "", "")

                result, run = self._run(args, child)
                self.assertEqual(result, supervisor.EXIT_SUPERVISOR_FAILCLOSED)
                self.assertEqual(run.call_count, 3)
                failure = self._failure(args)
                self.assertEqual(failure["failure_class"], "invalid_evidence")
                self.assertEqual(
                    failure["completed_arms"],
                    [
                        "expert_int4",
                        "expert_int4_channel_alpha_123",
                    ],
                )
                self.assertEqual(failure["invalid_arms"], ["expert_int4_channel_alpha"])
                self.assertEqual(failure["decision"]["decision"], 4)
                self.assertNotIn("ranking", failure["comparison"])

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
