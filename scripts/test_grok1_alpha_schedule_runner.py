"""Small subprocess and filesystem tests for fail-closed GH125 supervision."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path

# Bounded, shell-free fixture processes only.
import subprocess  # nosec B404
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_grok1_alpha_schedule_protocol import fixture

GIB = 1024**3


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("grok1_alpha_schedule_runner"),
            "bounded four-cell runner is missing",
        )
        import grok1_alpha_schedule_runner

        self.r = grok1_alpha_schedule_runner
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rss_handshakes = []
        read_process_rows = self.r._process_rows

        def sampled_process_rows():
            # Keep the real /proc snapshot and RSS values. Release only fixture
            # children that this exact snapshot has observed with positive RSS.
            rows = read_process_rows()
            for ready, sampled in self.rss_handshakes:
                try:
                    pid = int(ready.read_text())
                except (OSError, ValueError):
                    continue
                if rows.get(pid, {}).get("rss", 0) > 0:
                    sampled.touch()
            return rows

        observer = patch.object(self.r, "_process_rows", side_effect=sampled_process_rows)
        observer.start()
        self.addCleanup(observer.stop)

    def test_resource_boundaries_and_rejection_before_launch(self):
        good = {
            "mem_available": 24 * GIB,
            "swap_free": 2 * GIB,
            "competing": [],
            "disk_free": {"out": 40 * GIB + 123},
        }
        self.r.check_resources(good, {"out": 123})
        for field, value in (
            ("mem_available", 24 * GIB - 1),
            ("swap_free", 2 * GIB - 1),
            ("disk_free", {"out": 40 * GIB + 122}),
            ("competing", [17]),
        ):
            bad = {**good, field: value}
            out = self.root / field
            marker = self.root / "launched"
            code = f"from pathlib import Path; Path({str(marker)!r}).touch()"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = self.r.supervise(
                    out,
                    lambda: {},
                    lambda *a, code=code: [sys.executable, "-c", code],
                    lambda bad=bad: self.r.check_resources(bad, {"out": 123}),
                )
            self.assertNotEqual(result, 0)
            diagnostic = {
                "mem_available": "MemAvailable below 24 GiB",
                "swap_free": "free swap below 2 GiB",
                "disk_free": "filesystem out: need",
                "competing": "competing workload PIDs: [17]",
            }[field]
            self.assertIn(diagnostic, stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(marker.exists())
            self.assertEqual(
                json.loads((out / "outcome.json").read_text())["status"], "inconclusive"
            )

    def command(self, cell, out, run_id, context):
        data = fixture(cell)
        data["run_id"] = run_id
        return self.metrics_command(out, json.dumps(data))

    def metrics_command(self, out, raw):
        ready, sampled = out / "rss-ready", out / "rss-sampled"
        self.rss_handshakes.append((ready, sampled))
        # A bounded handshake, rather than a fixed sleep, makes even very fast
        # successful children observable when the CI supervisor is descheduled.
        return [
            sys.executable,
            "-c",
            "import os,time\nfrom pathlib import Path\n"
            f"Path({str(out / 'metrics.json')!r}).write_text({raw!r})\n"
            f"Path({str(ready)!r}).write_text(str(os.getpid()))\n"
            "deadline = time.monotonic() + 5\n"
            f"while not Path({str(sampled)!r}).exists():\n"
            "    if time.monotonic() >= deadline:\n"
            "        raise SystemExit('test fixture RSS acknowledgement timed out')\n"
            "    time.sleep(.01)\n",
        ]

    def test_fixture_rss_handshake(self):
        out = self.root / "handshake"
        out.mkdir()
        # Fixture argv is built internally; no shell or external command text.
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        proc = subprocess.Popen(  # nosec B603
            self.command("A", out, "fresh-run", {}),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5
            while not (out / "metrics.json").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((out / "metrics.json").exists())
            with self.assertRaises(subprocess.TimeoutExpired):
                proc.wait(timeout=0.1)
            self.assertFalse((out / "rss-sampled").exists())
            rows = self.r._process_rows()
            self.assertGreater(rows[proc.pid]["rss"], 0)
            self.assertTrue((out / "rss-sampled").exists())
            self.assertEqual(proc.wait(timeout=5), 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_complete_cells_publish(self):
        out = self.root / "success"
        self.assertEqual(self.r.supervise(out, lambda: {}, self.command, lambda: None), 0)
        data = json.loads((out / "outcome.json").read_text())
        self.assertEqual(data["status"], "complete")
        self.assertEqual(set(data["cells"]), set("ABCD"))
        self.assertEqual(data["paired_contrasts"]["chain_exit_drift"]["contrasts"]["B-A"], 0.0)
        self.assertGreater(data["cells"]["A"]["resources"]["process_tree_peak_rss_bytes"], 0)

    def test_report_columns_match_headers_after_cell_reordering_or_json_reload(self):
        from grok1_alpha_schedule_protocol import analyze
        from json_canonical import canonical_json

        for order, reload_json in (("DCBA", False), ("ABCD", True), ("DCBA", True)):
            cells = {c: fixture(c) for c in order}
            for cell, drift in {"A": 0.1, "B": 0.3, "C": 0.4, "D": 0.9}.items():
                cells[cell]["chain"]["end_of_chain"]["expert_only_chain_exit"][
                    "residual_drift_relative_norm"
                ] = drift
            payload = analyze(cells)
            if reload_json:
                payload = json.loads(canonical_json(payload))
            report = self.r.render_report(payload)
            with self.subTest(order=order, reload_json=reload_json):
                row = next(
                    line for line in report.splitlines() if line.startswith("| chain_exit_drift |")
                )
                self.assertEqual(
                    row,
                    "| chain_exit_drift | lower | 0.1 | 0.3 | 0.4 | 0.9 | 0.2 | 0.5 | 0.3 | 0.6 | 0.3 |",
                )

    def nonfinite_command(self, bad, raw_outputs):
        def command(cell, dest, run_id, context):
            data = fixture(cell)
            data["run_id"] = run_id
            data["chain"]["per_block"][0]["expert_only"]["router_margin_mean_observed"] = bad
            raw = json.dumps(data)
            raw_outputs[cell] = raw
            return self.metrics_command(dest, raw)

        return command

    def test_nonfinite_raw_child_metrics_are_preserved_but_never_published_complete(self):
        for label, bad in (
            ("nan", float("nan")),
            ("inf", float("inf")),
            ("minus-inf", float("-inf")),
        ):
            out = self.root / label
            raw_outputs = {}

            command = self.nonfinite_command(bad, raw_outputs)
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(nonfinite=label), redirect_stdout(stdout), redirect_stderr(stderr):
                code = self.r.supervise(out, lambda: {}, command, lambda: None)
                self.assertNotEqual(code, 0)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("non-finite", stderr.getvalue())
                self.assertIn(
                    "A.chain.per_block[0].expert_only.router_margin_mean_observed",
                    stderr.getvalue(),
                )
                outcome = json.loads((out / "outcome.json").read_text())
                self.assertEqual(outcome["status"], "inconclusive")
                self.assertEqual(outcome["cells"], {})
                self.assertNotIn("paired_contrasts", outcome)
                raw_path = next(out.glob("runs/*/A/metrics.json"))
                self.assertEqual(raw_path.read_text(), raw_outputs["A"])
                self.assertFalse((raw_path.parent / "validated.json").exists())

    def failure_command(self, failure, cell, dest, run_id):
        if failure == "missing":
            return [sys.executable, "-c", "print('missing output')"]
        if failure == "nonzero":
            return [sys.executable, "-c", "raise SystemExit(7)"]
        if failure == "kill":
            return [sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"]
        data = fixture(cell)
        if failure != "stale":
            data["run_id"] = run_id
        if failure == "terminal":
            data["chain"].pop("end_of_chain")
        raw = "{broken" if failure == "malformed" else json.dumps(data)
        return self.metrics_command(dest, raw)

    def test_failures_replace_stale_success_and_preserve_logs(self):
        for failure in ("missing", "malformed", "nonzero", "kill", "stale", "terminal"):
            out = self.root / failure
            out.mkdir()
            (out / "outcome.json").write_text('{"status":"complete"}')
            (out / "results.md").write_text("old success")

            def command(cell, dest, run_id, context, failure=failure):
                return self.failure_command(failure, cell, dest, run_id)

            with self.subTest(failure=failure):
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = self.r.supervise(out, lambda: {}, command, lambda: None)
                self.assertNotEqual(code, 0)
                diagnostic = {
                    "missing": "FileNotFoundError",
                    "malformed": "JSONDecodeError",
                    "nonzero": "child failure",
                    "kill": "child failure",
                    "stale": "stale run identity",
                    "terminal": "end_of_chain",
                }[failure]
                self.assertIn(diagnostic, stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                data = json.loads((out / "outcome.json").read_text())
                self.assertEqual(data["status"], "inconclusive")
                self.assertNotIn("paired_contrasts", data)
                self.assertNotIn("old success", (out / "results.md").read_text())
                self.assertTrue(list(out.glob("runs/*/A/child.log")))

    def test_timeout_kills_process_group_and_records_resources(self):
        log = self.root / "timeout.log"
        result = self.r.run_child([sys.executable, "-c", "import time; time.sleep(10)"], log, 0.15)
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["returncode"], 0)
        self.assertGreater(result["process_tree_peak_rss_bytes"], 0)

    def test_supervisor_interrupt_is_inconclusive(self):
        def gate():
            raise KeyboardInterrupt("fixture operator interrupt")

        out = self.root / "interrupt"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = self.r.supervise(out, lambda: {}, self.command, gate)
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("KeyboardInterrupt: fixture operator interrupt", stderr.getvalue())
        self.assertEqual(json.loads((out / "outcome.json").read_text())["status"], "inconclusive")

    def test_output_lock_and_historical_path_guard(self):
        out = self.root / "locked"
        with self.r.output_lock(out, nonblocking=True):
            with self.assertRaises(RuntimeError):
                self.r.supervise(out, lambda: {}, self.command, lambda: None)
        for name in ("grok-1-expert-precision-remedy-v4", "grok-1-expert-precision-remedy-v4/sub"):
            with self.assertRaises(ValueError):
                self.r.validate_output_path(self.root / name)

    def test_git_boundary_rejects_dirty_or_changed_source(self):
        commit = "a" * 40

        def git(*args):
            if args[0] == "rev-parse":
                return commit.encode()
            if args[0] == "status":
                return b""
            if args[0] == "show":
                return (self.r.REPO_ROOT / args[1].split(":", 1)[1]).read_bytes()
            raise AssertionError(args)

        with patch.object(self.r, "git_output", side_effect=git):
            self.assertEqual(self.r.clean_implementation()["commit"], commit)
        for bad_command in ("status", "show", "rev-parse"):

            def broken(*args, bad_command=bad_command):
                return b"dirty" if args[0] == bad_command else git(*args)

            with (
                self.subTest(bad_command=bad_command),
                patch.object(self.r, "git_output", side_effect=broken),
            ):
                with self.assertRaises(ValueError):
                    self.r.clean_implementation()

    def test_actual_shapes_define_payload_and_disk_costs(self):
        path = self.root / "weights.npy"
        np.save(path, np.ones((2, 3, 4), dtype=np.float32))
        record = self.r.tensor_inventory(path, 0)
        self.assertEqual(record["parameters"], 24)
        self.assertEqual(record["scale_parameters"], 8)
        self.assertEqual(record["source_payload_bytes"], 96)
        self.assertEqual(self.r.estimated_additional_bytes([record]), 24 + 2 * 32 + 4096)

    def test_actual_resource_files_and_hp_retention(self):
        inventory = []
        side = self.root / "side"
        side.mkdir()
        for b in range(4):
            path = self.root / f"block_{b:03d}__slot_00__moe_expert__gate.npy"
            np.save(path, np.ones((2, 3, 4), dtype=np.float32))
            inventory.append(self.r.tensor_inventory(path, b))
            for base in (side / f"block_{b:03d}", side / "ls-alpha" / f"block_{b:03d}"):
                base.mkdir(parents=True)
                np.save(base / f"{path.stem}__scale_f32.npy", np.ones((2, 4), dtype=np.float32))
            np.save(
                side / f"block_{b:03d}" / f"{path.stem}__q_int8.npy",
                np.ones((2, 3, 4), dtype=np.int8),
            )
        a = self.r.actual_resources(inventory, "A", side, self.root)
        d = self.r.actual_resources(inventory, "D", side, self.root)
        self.assertEqual(a["actual_code_payload_bytes"], 96)
        self.assertEqual(a["scale_payload_bytes"], 128)
        self.assertEqual(a["fp16_expert_payload_bytes"], 0)
        self.assertEqual(d["fp16_expert_payload_bytes"], 144)
        self.assertEqual(d["measured_expert_payload_bytes"], 200)

    def test_cli_help_and_preflight_failure_without_model_or_git(self):
        script = Path(__file__).with_name("grok1_alpha_schedule_ablation.py")
        self.assertTrue(script.exists(), "standalone CLI is missing")
        # Fixed local CLI help probe; no shell and no caller-supplied argv.
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        help_result = subprocess.run(  # nosec B603
            [sys.executable, str(script), "--help"], capture_output=True, text=True
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        import grok1_alpha_schedule_ablation as cli

        out = self.root / "preflight"
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(self.r, "git_output", side_effect=AssertionError("Git must not run")),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = cli.main(
                [
                    "--npy-root",
                    str(self.root / "missing"),
                    "--pack-root",
                    str(self.root / "missing"),
                    "--embedding-shard",
                    str(self.root / "missing.npy"),
                    "--out",
                    str(out),
                    "--preflight-only",
                ]
            )
        self.assertNotEqual(code, 0)
        diagnostic = json.loads(stdout.getvalue())
        self.assertEqual(diagnostic["status"], "inconclusive")
        self.assertIn("missing.npy", diagnostic["error"])
        self.assertFalse(diagnostic["model_forward_executed"])
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads((out / "preflight.json").read_text())["status"], "inconclusive")

    def test_real_supervisor_signal_reaps_running_child(self):
        out = self.root / "signal"
        pidfile = self.root / "child.pid"
        child = f"import os,time; from pathlib import Path; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(20)"
        program = (
            f"import sys; sys.path.insert(0, {str(Path(__file__).parent.resolve())!r}); "
            "import grok1_alpha_schedule_runner as r; from pathlib import Path; "
            f"raise SystemExit(r.supervise(Path({str(out)!r}), lambda: {{}}, "
            f"lambda *a: [sys.executable, '-c', {child!r}], lambda: None))"
        )
        # Internally generated signal fixture, executed directly without a shell.
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-c", program], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            deadline = time.monotonic() + 5
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(pidfile.exists())
            pid = int(pidfile.read_text())
            proc.terminate()
            self.assertNotEqual(proc.wait(timeout=5), 0)
            self.assertFalse(Path(f"/proc/{pid}").exists())
            self.assertEqual(
                json.loads((out / "outcome.json").read_text())["status"], "inconclusive"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_child_exit_with_live_grandchild_is_not_success(self):
        # A process leader exiting zero must not certify work still running in
        # its process tree; cleanup must terminate the orphaned descendant.
        code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])"
        result = self.r.run_child([sys.executable, "-c", code], self.root / "tree.log", 1)
        self.assertTrue(result.get("unfinished_descendants", False))
        self.assertNotEqual(result["returncode"], 0)

    def write_small_inputs(self):
        npy = self.root / "npy"
        packs = self.root / "packs"
        packs.mkdir()
        for b in range(4):
            folder = npy / f"goz68-block_{b:03d}-attn"
            folder.mkdir(parents=True)
            for i, role in enumerate(("gate", "down", "up")):
                np.save(
                    folder / f"block_{b:03d}__slot_{i:02d}__moe_expert__{role}.npy",
                    np.ones((2, 3, 4), dtype=np.float32),
                )
            (packs / f"block_{b:03d}-attention_plus_expert.goz1").write_bytes(
                b"synthetic pack identity"
            )
        embedding = self.root / "embedding.npy"
        np.save(embedding, np.ones((1, 6144), dtype=np.float32))
        return npy, packs, embedding

    def cell_fixture(self, cli):
        npy, packs, embedding = self.write_small_inputs()
        side = self.root / "side"
        out = self.root / "cell"
        out.mkdir()
        args = cli.build_parser().parse_args(
            [
                "--npy-root",
                str(npy),
                "--pack-root",
                str(packs),
                "--embedding-shard",
                str(embedding),
                "--out",
                str(out),
                "--cell",
                "C",
                "--manifest",
                str(out / "launch.json"),
            ]
        )
        paths = cli.paths_for(args)
        inventory = cli.inspect_inputs(paths, out, side)
        return args, paths, out, side, inventory

    def chain_forward(self, inventory, side, context):
        def forward(blocks, chain_paths, **kwargs):
            self.assertEqual(blocks, [0, 1, 2, 3])
            self.assertEqual(
                (kwargs["tokens"], kwargs["seed"], kwargs["top_k"]), (8192, 20260806, 2)
            )
            self.assertEqual(kwargs["hp_blocks"], {1, 2, 3})
            self.assertEqual(kwargs["expert_mode"], "int4")
            self.assertFalse(kwargs["skip_fp16"])
            for row in inventory:
                if row["block"] == 0:
                    base = side / "block_000"
                    base.mkdir(parents=True, exist_ok=True)
                    stem = Path(row["path"]).stem
                    np.save(base / f"{stem}__q_int8.npy", np.ones((2, 3, 4), dtype=np.int8))
                    np.save(base / f"{stem}__scale_f32.npy", np.ones((2, 4), dtype=np.float32))
            chain = fixture("C")["chain"]
            for actual, expected in zip(chain["pack_provenance"], context["inputs"], strict=True):
                actual.update(expected)
            return chain

        return forward

    def test_cell_entry_uses_frozen_chain_arguments_and_real_content_binding(self):
        import grok1_alpha_schedule_ablation as cli

        args, paths, out, side, inventory = self.cell_fixture(cli)

        def git(*arguments):
            if arguments[0] == "rev-parse":
                return b"a" * 40
            if arguments[0] == "status":
                return b""
            return (self.r.REPO_ROOT / arguments[1].split(":", 1)[1]).read_bytes()

        def snapshot(requirements):
            return {
                "mem_available": 24 * GIB,
                "swap_free": 2 * GIB,
                "competing": [],
                "disk_free": {k: 40 * GIB + r["additional"] for k, r in requirements.items()},
            }

        with (
            patch.object(self.r, "git_output", side_effect=git),
            patch.object(self.r, "resource_snapshot", side_effect=snapshot),
        ):
            context = cli.prepare_context(paths, inventory)
            (out / "launch.json").write_text(
                json.dumps(
                    {
                        "parent_pid": os.getppid(),
                        "cell": "C",
                        "run_id": "fresh-run",
                        "context": context,
                    }
                )
            )

            forward = self.chain_forward(inventory, side, context)
            with patch.object(cli, "run_chain", side_effect=forward):
                self.assertEqual(cli.run_cell(args, paths, out, side), 0)
        data = json.loads((out / "metrics.json").read_text())
        self.assertEqual(data["provenance"]["token_ids"][0], 37)
        self.assertEqual(data["provenance"]["token_ids"][-1], 131061)
        self.assertEqual(data["resources"]["actual_code_payload_bytes"], 72)
        self.assertEqual(data["resources"]["fp16_expert_payload_bytes"], 432)


if __name__ == "__main__":
    unittest.main()
