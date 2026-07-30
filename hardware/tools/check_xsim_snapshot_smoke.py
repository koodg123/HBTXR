#!/usr/bin/env python3
"""Run a minimal Vivado XSIM snapshot smoke test.

The HGTXR HLS RTL cosim currently reaches xelab snapshot generation, then
fails during the XSIM Tcl snapshot-load step. This tool separates design
failures from host XSIM runtime failures by compiling and running a trivial
Verilog testbench with the same Vivado installation.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "xsim_snapshot_smoke_2026_06_29.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "xsim_snapshot_smoke_2026_06_29.md"
DEFAULT_XILINX_ROOT = Path("/tools/Xilinx")
PASS_TOKEN = "XSIM_SMOKE_PASS"


def vivado_env(xilinx_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    lib_paths = [
        xilinx_root / "Vivado" / "2023.2" / "lib" / "lnx64.o" / "Ubuntu" / "22",
        xilinx_root / "Vivado" / "2023.2" / "lib" / "lnx64.o" / "Ubuntu",
        xilinx_root / "Vivado" / "2023.2" / "lib" / "lnx64.o" / "Rhel" / "9",
        xilinx_root / "Vivado" / "2023.2" / "lib" / "lnx64.o",
        xilinx_root / "Vitis_HLS" / "2023.2" / "lib" / "lnx64.o" / "Ubuntu" / "22",
        xilinx_root / "Vitis_HLS" / "2023.2" / "lib" / "lnx64.o" / "Ubuntu",
        xilinx_root / "Vitis_HLS" / "2023.2" / "lib" / "lnx64.o" / "Rhel" / "9",
        Path("/usr/lib/x86_64-linux-gnu"),
    ]
    existing = [str(path) for path in lib_paths if path.exists()]
    if env.get("LD_LIBRARY_PATH"):
        existing.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(existing)
    env["TERM"] = "xterm"
    env["TERMINFO"] = "/lib/terminfo"
    return env


def run_command(cmd: Sequence[str], cwd: Path, env: dict[str, str], timeout_s: int) -> dict[str, Any]:
    proc = subprocess.run(
        list(cmd),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    return {
        "cmd": list(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
    }


def run_command_with_input(
    cmd: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    stdin_text: str,
    timeout_s: int,
    input_delay_s: float = 0.0,
) -> dict[str, Any]:
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=env,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if input_delay_s:
        time.sleep(input_delay_s)
    stdout, _ = proc.communicate(stdin_text, timeout=timeout_s)
    return {
        "cmd": list(cmd),
        "returncode": proc.returncode,
        "stdin": stdin_text,
        "stdout": stdout,
    }


def run_xsimk_mi_sequence(cmd: Sequence[str], cwd: Path, env: dict[str, str], timeout_s: int) -> dict[str, Any]:
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=env,
        text=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    os.set_blocking(proc.stdout.fileno(), False)

    output = bytearray()
    stdin_log: list[str] = []
    sent_run = False
    sent_continue = False
    sent_exit = False
    start = time.monotonic()
    deadline = time.monotonic() + timeout_s
    timed_out = False

    def write_cmd(line: str) -> None:
        stdin_log.append(line)
        proc.stdin.write(line.encode("utf-8"))
        proc.stdin.flush()

    while True:
        if proc.poll() is not None:
            break
        if time.monotonic() > deadline:
            timed_out = True
            proc.kill()
            proc.wait(timeout=5)
            break

        readable, _, _ = select.select([proc.stdout], [], [], 0.1)
        if readable:
            try:
                chunk = os.read(proc.stdout.fileno(), 4096)
            except BlockingIOError:
                chunk = b""
            if chunk:
                output.extend(chunk)

        text = output.decode("utf-8", errors="replace")
        if not sent_run and "elaboration-done" in text:
            write_cmd("-exec-run\n")
            sent_run = True
        elif sent_run and not sent_continue and 'reason="run-complete"' in text:
            write_cmd("-exec-continue\n")
            sent_continue = True
        elif sent_continue and not sent_exit and 'reason="finish"' in text:
            write_cmd("-gdb-exit\n")
            sent_exit = True

    if proc.stdout is not None:
        while True:
            try:
                chunk = os.read(proc.stdout.fileno(), 4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            output.extend(chunk)

    return {
        "cmd": list(cmd),
        "returncode": proc.returncode,
        "stdin": "".join(stdin_log),
        "stdout": output.decode("utf-8", errors="replace"),
        "timed_out": timed_out,
        "elapsed_s": round(time.monotonic() - start, 3),
    }


def run_smoke(xilinx_root: Path, keep_workdir: Path | None, timeout_s: int) -> dict[str, Any]:
    workdir = keep_workdir or Path(tempfile.mkdtemp(prefix="hgtxr_xsim_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)
    tb_path = workdir / "tb.v"
    tb_path.write_text(
        "\n".join(
            [
                "module tb;",
                "  initial begin",
                f'    $display("{PASS_TOKEN}");',
                "    $finish;",
                "  end",
                "endmodule",
                "",
            ]
        )
    )
    run_all_tcl = workdir / "run_all_no_wave.tcl"
    run_all_tcl.write_text("run all\nquit\n")

    env = vivado_env(xilinx_root)
    vivado_bin = xilinx_root / "Vivado" / "2023.2" / "bin"
    steps = [
        run_command([str(vivado_bin / "xvlog"), "tb.v"], workdir, env, timeout_s),
        run_command([str(vivado_bin / "xelab"), "tb", "-s", "tb_snapshot"], workdir, env, timeout_s),
        run_command([str(vivado_bin / "xsim"), "tb_snapshot", "-R"], workdir, env, timeout_s),
    ]
    alternate_xsim_steps = [
        run_command([str(vivado_bin / "xsim"), "tb_snapshot", "--runall"], workdir, env, timeout_s),
        run_command([str(vivado_bin / "xsim"), "tb_snapshot", "--tclbatch", str(run_all_tcl)], workdir, env, timeout_s),
    ]
    xsimk_path = workdir / "xsim.dir" / "tb_snapshot" / "xsimk"
    direct_xsimk_step = (
        run_xsimk_mi_sequence(
            [str(xsimk_path)],
            workdir,
            env,
            timeout_s,
        )
        if xsimk_path.exists()
        else {"cmd": [str(xsimk_path)], "returncode": None, "stdout": "xsimk not found"}
    )
    xsim_outputs = [steps[-1]["stdout"], *[step["stdout"] for step in alternate_xsim_steps], direct_xsimk_step["stdout"]]
    xsim_log = workdir / "xsim.log"
    xsim_log_text = xsim_log.read_text(errors="ignore") if xsim_log.exists() else ""
    snapshot_built = "Built simulation snapshot tb_snapshot" in steps[1]["stdout"]
    direct_xsimk_pass = PASS_TOKEN in direct_xsimk_step["stdout"]
    smoke_pass = any(PASS_TOKEN in output for output in xsim_outputs) or PASS_TOKEN in xsim_log_text
    xsim_launch_exception = any("unexpected exception when evaluating tcl command" in output for output in xsim_outputs) or (
        "unexpected exception when evaluating tcl command" in xsim_log_text
    )
    autoloadwcfg_in_failure = any("-autoloadwcfg" in output for output in xsim_outputs) or "-autoloadwcfg" in xsim_log_text

    if smoke_pass and xsim_launch_exception:
        status = "wrapper-blocked-kernel-pass"
    elif smoke_pass:
        status = "pass"
    elif snapshot_built and xsim_launch_exception:
        status = "blocked-xsim-runtime"
    else:
        status = "fail"
    return {
        "status": status,
        "workdir": str(workdir),
        "xilinx_root": str(xilinx_root),
        "checks": {
            "snapshot_built": snapshot_built,
            "smoke_pass_token_seen": smoke_pass,
            "xsim_launch_exception": xsim_launch_exception,
            "autoloadwcfg_in_failure": autoloadwcfg_in_failure,
            "alternate_xsim_attempted": True,
            "alternate_xsim_pass_token_seen": any(PASS_TOKEN in step["stdout"] for step in alternate_xsim_steps),
            "direct_xsimk_attempted": xsimk_path.exists(),
            "direct_xsimk_entered_kernel": "elaboration-done" in direct_xsimk_step["stdout"],
            "direct_xsimk_exec_run_complete": 'reason="run-complete"' in direct_xsimk_step["stdout"],
            "direct_xsimk_exec_continue_finish": 'reason="finish"' in direct_xsimk_step["stdout"],
            "direct_xsimk_pass_token_seen": direct_xsimk_pass,
        },
        "steps": steps,
        "alternate_xsim_steps": alternate_xsim_steps,
        "direct_xsimk_step": direct_xsimk_step,
        "sources": {
            "tb": str(tb_path),
            "run_all_tcl": str(run_all_tcl),
            "xsimk": str(xsimk_path),
            "xsim_log": str(xsim_log),
            "xsim_script": str(workdir / "xsim.dir" / "tb_snapshot" / "xsim_script.tcl"),
        },
        "interpretation": (
            "Minimal XSIM snapshot execution works through the normal xsim wrapper."
            if status == "pass"
            else "Minimal Verilog snapshot builds and direct xsimk MI execution runs the testbench, "
            "but the normal xsim wrapper/Tcl launch path still fails. HLS cosim cannot be signed off "
            "through the standard wrapper until that path is repaired or bypassed."
            if status == "wrapper-blocked-kernel-pass"
            else "Minimal Verilog snapshot builds, but XSIM fails during snapshot launch. "
            "This indicates a host XSIM runtime issue independent of the HGTXR RTL."
            if status == "blocked-xsim-runtime"
            else "Minimal XSIM smoke failed before proving snapshot launch."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# XSIM Snapshot Smoke",
        "",
        f"- status: `{result['status']}`",
        f"- workdir: `{result['workdir']}`",
        f"- Xilinx root: `{result['xilinx_root']}`",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "|---|---:|",
    ]
    for name, value in sorted(result["checks"].items()):
        lines.append(f"| `{name}` | `{value}` |")
    lines.extend(["", "## Interpretation", "", result["interpretation"], "", "## Sources", ""])
    for name, path in sorted(result["sources"].items()):
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal Vivado XSIM snapshot smoke test.")
    parser.add_argument("--xilinx-root", type=Path, default=DEFAULT_XILINX_ROOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--workdir", type=Path, default=None, help="Keep/use this workdir instead of a tempdir.")
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--cleanup", action="store_true", help="Remove temp workdir after writing reports.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(args.xilinx_root.resolve(), args.workdir, args.timeout_s)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.cleanup and args.workdir is None:
        shutil.rmtree(result["workdir"], ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
