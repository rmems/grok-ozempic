"""Startup, preflight and cleanup boundary regressions for GH125."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess  # nosec B404  # Timeout exception and mock boundary only.
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_grok1_alpha_schedule_runner import write_small_inputs

GIB = 1024**3


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        import grok1_alpha_schedule_runner

        self.r = grok1_alpha_schedule_runner
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_initial_publication_interrupt_replaces_stale_success(self):
        out = self.root / "early-interrupt"
        out.mkdir()
        self.r.atomic_json(out / "outcome.json", {"status": "complete"})
        original = self.r.atomic_json
        calls = 0

        def interrupted_write(path, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt("before authoritative replacement")
            original(path, payload)

        with patch.object(self.r, "atomic_json", side_effect=interrupted_write):
            self.assertEqual(self.r.supervise(out, Mock(), Mock(), Mock()), 1)
        outcome = json.loads((out / "outcome.json").read_text())
        self.assertEqual(outcome["status"], "inconclusive")
        self.assertIn("before authoritative replacement", outcome["error"])

    def test_cleanup_timeout_preserves_original_failure(self):
        proc = Mock(pid=987654321)
        proc.wait.side_effect = subprocess.TimeoutExpired("fixture child", 10)
        stderr = io.StringIO()
        with (
            patch.object(self.r.subprocess, "Popen", return_value=proc),
            patch.object(self.r, "_kill_group"),
            patch.object(self.r, "_group_rss", side_effect=ValueError("original sampling failure")),
            redirect_stderr(stderr),
            self.assertRaisesRegex(ValueError, "original sampling failure"),
        ):
            self.r.run_child(["fixture"], self.root / "cleanup.log")
        self.assertIn("cleanup", stderr.getvalue())
        self.assertIn("TimeoutExpired", stderr.getvalue())

    def test_run_identity_interrupt_replaces_stale_success(self):
        out = self.root / "identity-interrupt"
        out.mkdir()
        self.r.atomic_json(out / "outcome.json", {"status": "complete"})
        with patch.object(self.r.uuid, "uuid4", side_effect=KeyboardInterrupt("identity setup")):
            try:
                code = self.r.supervise(out, Mock(), Mock(), Mock())
            except KeyboardInterrupt:
                self.fail("run identity setup escaped the guarded lifecycle")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((out / "outcome.json").read_text())["status"], "inconclusive")

    def test_startup_interrupts_replace_stale_success_and_restore_handlers(self):
        for boundary in ("mkdir", "handler"):
            with self.subTest(boundary=boundary):
                self.check_startup_interrupt(boundary)

    def check_startup_interrupt(self, boundary):
        out = self.root / boundary
        out.mkdir()
        self.r.atomic_json(out / "outcome.json", {"status": "complete"})
        old = self.r.signal.getsignal(self.r.signal.SIGTERM)
        real_mkdir, real_signal = Path.mkdir, self.r.signal.signal
        fired = False

        def mkdir(path, *args, **kwargs):
            nonlocal fired
            if boundary == "mkdir" and path == out and not fired:
                fired = True
                raise KeyboardInterrupt("startup mkdir")
            return real_mkdir(path, *args, **kwargs)

        def install(signum, handler):
            nonlocal fired
            result = real_signal(signum, handler)
            if boundary == "handler" and not fired:
                fired = True
                raise KeyboardInterrupt("startup handler")
            return result

        with patch.object(Path, "mkdir", mkdir), patch.object(self.r.signal, "signal", install):
            self.assertEqual(self.r.supervise(out, Mock(), Mock(), Mock()), 1)
        self.assertEqual(self.r.signal.getsignal(self.r.signal.SIGTERM), old)
        self.assertEqual(json.loads((out / "outcome.json").read_text())["status"], "inconclusive")

    def test_cleanup_timeout_without_original_failure_is_not_suppressed(self):
        proc = Mock(pid=987654321, returncode=0)
        proc.wait.side_effect = subprocess.TimeoutExpired("fixture child", 10)
        with (
            patch.object(self.r.subprocess, "Popen", return_value=proc),
            patch.object(self.r, "_kill_group"),
            patch.object(self.r, "_group_rss", return_value=1),
            patch.object(self.r, "_unfinished_descendants", return_value=False),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            self.r.run_child(["fixture"], self.root / "cleanup.log")

    def test_launcher_cache_and_cell_scratch_are_disjoint(self):
        import grok1_alpha_schedule_ablation as cli

        out = self.root / "report"
        side = out / "int4-side"
        cell_out = out / "runs" / "fresh-run" / "A"
        side.mkdir(parents=True)
        cell_out.mkdir(parents=True)
        (side / "cache.bin").write_bytes(b"123456789")
        args = cli.build_parser().parse_args(
            [
                "--npy-root",
                str(self.root / "npy"),
                "--pack-root",
                str(self.root / "packs"),
                "--embedding-shard",
                str(self.root / "embedding.npy"),
                "--out",
                str(out),
            ]
        )
        command = cli.make_command(args, side)("A", cell_out, "fresh-run", {})
        child_args = cli.build_parser().parse_args(command[2:])
        self.assertEqual(child_args.out, cell_out)
        self.assertEqual(child_args.int4_side_root, side)
        resources = self.r.actual_resources([], "A", child_args.int4_side_root, child_args.out)
        self.assertEqual(resources["cache_disk_bytes"], 9)
        self.assertEqual(resources["scratch_disk_bytes"], (cell_out / "launch.json").stat().st_size)

    def test_published_preflight_drops_machine_local_paths(self):
        import grok1_alpha_schedule_ablation as cli

        npy, packs, embedding = write_small_inputs(self.root)
        out = self.root / "preflight-private"
        args = [
            "--npy-root",
            str(npy),
            "--pack-root",
            str(packs),
            "--embedding-shard",
            str(embedding),
            "--out",
            str(out),
            "--preflight-only",
        ]

        def snapshot(requirements):
            return {
                "mem_available": 24 * GIB,
                "swap_free": 2 * GIB,
                "competing": [],
                "disk_free": {k: 40 * GIB + r["additional"] for k, r in requirements.items()},
            }

        with (
            patch.object(self.r, "resource_snapshot", side_effect=snapshot),
            patch.object(
                self.r, "clean_implementation", return_value={"commit": "a" * 40, "dirty": False}
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cli.main(args), 0)
        for name in ("preflight.json", "resource-preflight.json"):
            self.assertNotIn(str(self.root), (out / name).read_text())
        self.assertEqual(len(json.loads((out / "preflight.json").read_text())["inventory"]), 12)

    def test_source_overlap_rejected_without_any_writes(self):
        import grok1_alpha_schedule_ablation as cli

        for mode in ([], ["--preflight-only"]):
            for target in ("output", "cache"):
                with self.subTest(mode=mode, target=target):
                    source = self.root / "sources"
                    source.mkdir(exist_ok=True)
                    overlap = source / "goz68-block_000-attn" / "invalid-output"
                    args = [
                        "--npy-root",
                        str(source),
                        "--pack-root",
                        str(source),
                        "--embedding-shard",
                        str(source / "missing.npy"),
                        "--out",
                        str(overlap if target == "output" else self.root / "safe-out"),
                    ]
                    if target == "cache":
                        args.extend(["--int4-side-root", str(overlap)])
                    before = sorted(self.root.rglob("*"))
                    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                        self.assertEqual(cli.main(args + mode), 1)
                    self.assertEqual(sorted(self.root.rglob("*")), before)


if __name__ == "__main__":
    unittest.main()
