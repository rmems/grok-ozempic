"""Bounded Linux GH125 execution and restart-safe evidence publication."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
# Capture before importing execution dependencies; every local Python source is
# pinned to the clean commit, including the new CLI, protocol and cache helpers.
_SOURCE_BYTES = {str(p.relative_to(REPO_ROOT)): p.read_bytes()
                 for p in (REPO_ROOT / "scripts").glob("*.py") if not p.name.startswith("test_")}

import numpy as np  # noqa: E402

from grok1_alpha_schedule_protocol import (  # noqa: E402
    CELLS, CONTRAST_KEYS, PROTOCOL, RESOURCE_SCOPE, analyze, require, validate_cell,
)
# Reuse durable publication and locking, without importing GH85 decision logic
# into the protocol or replacing any historical supervisor functions.
from grok1_multiblock_v4_supervisor import (  # noqa: E402
    _atomic_write_json as atomic_json,
    _atomic_write_text as atomic_text,
    _supervisor_output_lock as output_lock,
)

GIB = 1024**3
POLL_SECONDS = .05
CHILD_TIMEOUT_SECONDS = 24*60*60


def validate_output_path(path):
    resolved = Path(path).expanduser().resolve()
    require(not any(p.startswith("grok-1-expert-precision-remedy") for p in resolved.parts),
            "output/cache collides with historical precision-remedy evidence")
    require(resolved != REPO_ROOT and resolved != REPO_ROOT.parent, "output must be a dedicated directory")
    return resolved


def git_output(*arguments):
    """The only Git I/O boundary; tests substitute this function, never its checks."""
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), *arguments], capture_output=True,
        timeout=30, check=True).stdout  # noqa: S607


def clean_implementation():
    commit = git_output("rev-parse", "HEAD").decode("ascii").strip()
    require(re.fullmatch("[0-9a-f]{40}", commit) is not None, "invalid implementation commit")
    require(not git_output("status", "--porcelain", "--untracked-files=all", "--", "scripts", "src"),
            "implementation has uncommitted sources")
    for relative, resident in _SOURCE_BYTES.items():
        require(git_output("show", f"{commit}:{relative}") == resident
                == (REPO_ROOT/relative).read_bytes(), f"implementation source changed: {relative}")
    require(git_output("rev-parse", "HEAD").decode("ascii").strip() == commit,
            "implementation changed while pinning")
    return {"commit": commit, "dirty": False}


def tensor_inventory(path, block):
    """Read NPY headers via mmap, without materializing tensor payloads."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    require(array.ndim == 3 and array.dtype == np.dtype("float32") and all(array.shape),
            f"invalid reference expert tensor: {path}")
    result = {"path": str(Path(path).resolve()), "block": block, "shape": list(array.shape),
              "source_dtype": str(array.dtype), "parameters": int(array.size),
              "scale_parameters": int(array.shape[0]*array.shape[-1]),
              "source_payload_bytes": int(array.nbytes), "source_file_bytes": Path(path).stat().st_size}
    del array
    return result


def estimated_additional_bytes(inventory):
    # Both scale modes share int8 codes. Reserve whole new generations even for
    # populated caches: interruption/rebinding may require their replacement.
    return sum(r["parameters"] + 8*r["scale_parameters"] + 4096 for r in inventory)


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def disk_requirements(out, side_root, inventory):
    amounts = {}
    for path, additional in ((out, 64*1024**2), (side_root, estimated_additional_bytes(inventory))):
        parent = existing_parent(path)
        device = str(parent.stat().st_dev)
        if device not in amounts:
            amounts[device] = {"path": str(parent), "additional": 0}
        amounts[device]["additional"] += additional
    return amounts


def _process_rows():
    rows = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            parts = path.read_text().rsplit(")", 1)[1].split()
            rows[int(path.parent.name)] = {"parent": int(parts[1]), "group": int(parts[2]),
                                           "rss": max(0, int(parts[21])) * os.sysconf("SC_PAGE_SIZE")}
        except (OSError, ValueError, IndexError):
            continue  # Process exited between enumeration and read.
    return rows


def resource_snapshot(requirements):
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, raw = line.partition(":")
        if key in ("MemAvailable", "SwapFree"):
            values[key] = int(raw.split()[0])*1024
    rows = _process_rows()
    ignored = {os.getpid(), os.getppid()}
    competing = [pid for pid, row in rows.items() if pid not in ignored and row["rss"] >= 2*GIB]
    return {"mem_available": values.get("MemAvailable", 0), "swap_free": values.get("SwapFree", 0),
            "competing": competing,
            "disk_free": {key: shutil.disk_usage(row["path"]).free for key, row in requirements.items()}}


def check_resources(snapshot, requirements):
    require(snapshot["mem_available"] >= 24*GIB, "MemAvailable below 24 GiB")
    require(snapshot["swap_free"] >= 2*GIB, "free swap below 2 GiB")
    require(not snapshot["competing"], f"competing workload PIDs: {snapshot['competing']}")
    for key, additional in requirements.items():
        require(snapshot["disk_free"][key] >= 40*GIB+additional,
                f"filesystem {key}: need 40 GiB plus {additional} additional bytes")


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_child(command, log, timeout=CHILD_TIMEOUT_SECONDS):
    """Sample aggregate RSS of the new process group and always reap its leader."""
    started, peak, timed_out, unfinished = time.monotonic(), 0, False, False
    with Path(log).open("wb") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,  # noqa: S603
                                start_new_session=True)
        try:
            while True:
                peak = max(peak, sum(r["rss"] for r in _process_rows().values()
                                     if r["group"] == proc.pid))
                if proc.poll() is not None:
                    unfinished = any(pid != proc.pid and row["group"] == proc.pid
                                     and row["rss"] > 0 for pid, row in _process_rows().items())
                    break
                if time.monotonic()-started >= timeout:
                    timed_out = True
                    break
                time.sleep(POLL_SECONDS)
        finally:
            # Includes grandchildren when a child exits early, on timeout, and
            # operator interruption. The fixed Linux session owns no other job.
            _kill_group(proc.pid)
            proc.wait(timeout=10)
    return {"returncode": proc.returncode if not unfinished else -signal.SIGKILL,
            "leader_returncode": proc.returncode, "unfinished_descendants": unfinished, "timed_out": timed_out,
            "wall_seconds": time.monotonic()-started, "process_tree_peak_rss_bytes": peak,
            "rss_sampling_interval_seconds": POLL_SECONDS}


@contextmanager
def interrupt_handlers():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"supervisor signal {signum}")
    old = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        yield
    finally:
        for signum, handler in old.items():
            signal.signal(signum, handler)


def render_report(payload):
    lines = ["# Grok-1 alpha/schedule ablation", "", f"Status: {payload['status']}",
             f"Run: {payload['run_id']}", "", "The authoritative record is outcome.json.", ""]
    if payload["status"] != "complete":
        return "\n".join(lines + ["No causal conclusion: incomplete evidence.", str(payload.get("error", "running")), ""])
    lines += [payload["interpretation"], "", payload["precision_cost_note"], "",
              payload["control_semantics"], "", "Top-1 0.95 remains diagnostic only.", "",
              "| Metric | Favorable | A | B | C | D | B-A | D-C | C-A | D-B | Interaction |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in payload["paired_contrasts"].items():
        vals = [row["values"][cell] for cell in CELLS]
        deltas = [row["contrasts"][key] for key in CONTRAST_KEYS] if row["contrasts"] else [None]*len(CONTRAST_KEYS)
        formatted = ["empty band" if v is None else f"{v:.9g}" for v in vals+deltas]
        lines.append("| " + " | ".join([name, row["favorable_direction"], *formatted]) + " |")
    lines += ["", "Observed attribution by metric (signs describe this run, not statistical significance):", ""]
    for name, row in payload["paired_contrasts"].items():
        if row.get("observed_effects"):
            lines.append(f"- {name}: " + "; ".join(f"{k}: {row['observed_effects'][k]}" for k in CONTRAST_KEYS))
    lines += ["", "Raw per-block controls, source identities and resource accounting are in outcome.json.", ""]
    return "\n".join(lines)


def supervise(out, prepare, command, gate):
    """Run four fresh children; callbacks bind the CLI's input/preflight boundary."""
    out = validate_output_path(out)
    with output_lock(out, nonblocking=True), interrupt_handlers():
        out.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        run_dir = out/"runs"/run_id
        run_dir.mkdir(parents=True)
        payload = {"protocol": PROTOCOL, "run_id": run_id, "status": "inconclusive",
                   "error": "run started; not all cells validated", "cells": {}}
        def publish():
            # JSON is authoritative. Writing inconclusive first invalidates any
            # old success before expensive preparation or model work starts.
            atomic_json(out/"outcome.json", payload)
            atomic_text(out/"results.md", render_report(payload))
            atomic_json(run_dir/"outcome.json", payload)
        publish()
        try:
            gate()
            context = prepare()
            for cell in CELLS:
                dest = run_dir/cell
                dest.mkdir()
                gate()
                observation = run_child(command(cell, dest, run_id, context), dest/"child.log")
                atomic_json(dest/"execution.json", observation)
                require(observation["returncode"] == 0 and not observation["timed_out"],
                        f"cell {cell} child failure: {observation}")
                evidence = json.loads((dest/"metrics.json").read_text())
                evidence["resources"].update({k: observation[k] for k in (
                    "wall_seconds", "process_tree_peak_rss_bytes", "rss_sampling_interval_seconds")})
                validate_cell(evidence, cell, run_id)
                if context.get("provenance"):
                    require(evidence["provenance"] == context["provenance"], "child differs from launch identities")
                payload["cells"][cell] = evidence
                atomic_json(dest/"validated.json", evidence)
                publish()
            payload = analyze(payload["cells"])
            publish()
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            payload.update(status="inconclusive", error=f"{type(exc).__name__}: {exc}")
            payload.pop("paired_contrasts", None)
            publish()
            print(payload["error"], file=sys.stderr)
            return 1


def disk_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def actual_resources(inventory, cell, side_root, out):
    """Separate actual NPY storage from logical INT4 and hypothetical packing."""
    spec = CELLS[cell]
    qcount = qfiles = scales = scalefiles = fp16 = nonexpert = 0
    expert_paths = {Path(r["path"]) for r in inventory}
    for folder in {p.parent for p in expert_paths}:
        for path in folder.glob("*.npy"):
            if path not in expert_paths:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                nonexpert += array.nbytes
                del array
    records = []
    for row in inventory:
        if row["block"] in spec.hp:
            fp16 += row["parameters"] * np.dtype("float16").itemsize
            records.append({**row, "representation": "FP16 weight roundtrip; FP32 compute"})
            continue
        stem = Path(row["path"]).stem
        base = Path(side_root)/f"block_{row['block']:03d}"
        scale_base = Path(side_root)/"ls-alpha"/base.name if spec.channel_alpha else base
        qpath, spath = base/f"{stem}__q_int8.npy", scale_base/f"{stem}__scale_f32.npy"
        q, s = (np.load(p, mmap_mode="r", allow_pickle=False) for p in (qpath, spath))
        require(q.dtype == np.dtype("int8") and list(q.shape) == row["shape"], "invalid actual code storage")
        require(s.dtype == np.dtype("float32") and list(s.shape) == [row["shape"][0], row["shape"][-1]],
                "invalid actual scale storage")
        qcount += q.size
        scales += s.nbytes
        qfiles += qpath.stat().st_size
        scalefiles += spath.stat().st_size
        records.append({**row, "representation": "int8 codes / float32 scales",
                        "code_file_bytes": qpath.stat().st_size, "scale_file_bytes": spath.stat().st_size})
        del q, s
    return {"quantized_parameters": int(qcount), "logical_code_bits": int(qcount*4),
            "code_dtype": "int8", "actual_code_payload_bytes": int(qcount),
            "code_file_bytes": qfiles, "scale_payload_bytes": int(scales), "scale_file_bytes": scalefiles,
            "fp16_expert_payload_bytes": fp16, "measured_expert_payload_bytes": int(qcount+scales+fp16),
            "high_precision_nonexpert_payload_bytes": int(nonexpert),
            "measured_block_payload_bytes": int(qcount+scales+fp16+nonexpert),
            "hypothetical_nibble_code_bytes": int((qcount+1)//2), "cache_disk_bytes": disk_bytes(side_root),
            "scratch_disk_bytes": disk_bytes(out), "scope": RESOURCE_SCOPE, "tensors": records,
            "wall_seconds": 0., "process_tree_peak_rss_bytes": 0,
            "rss_sampling_interval_seconds": POLL_SECONDS}
