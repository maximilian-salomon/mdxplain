#!/usr/bin/env python
"""Run all benchmark scripts and package their JSON outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
from typing import List


BENCHMARK_SCRIPTS = [
    Path("dev_scripts/benchmark_fast_standard.py"),
    Path("dev_scripts/benchmark_iterative.py"),
    Path("dev_scripts/benchmark_standard_full.py"),
]


def _run_command(cmd: List[str], cwd: Path) -> int:
    """Execute command and print runtime + return code."""
    print(f"\n=== Running: {' '.join(cmd)}")
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.perf_counter() - started
    print(f"=== Exit code: {result.returncode} (elapsed: {elapsed:.1f}s)")
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark scripts (optionally data generation first) and "
            "package all benchmark JSON outputs into one zip."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable to use for running scripts (default: current Python).",
    )
    parser.add_argument(
        "--include-generate-data",
        action="store_true",
        help="Run dev_scripts/benchmark_generate_data.py before benchmarks.",
    )
    parser.add_argument(
        "--skip-pack",
        action="store_true",
        help="Skip final JSON zip packaging step.",
    )
    parser.add_argument(
        "--pack-output",
        type=Path,
        default=None,
        help=(
            "Output zip path for JSON packaging. Default: "
            "benchmark_jsons_<timestamp>.zip in project root."
        ),
    )
    parser.add_argument(
        "--pack-benchmark-script",
        action="append",
        dest="pack_benchmark_scripts",
        help=(
            "Script path passed to pack_benchmark_jsons.py as --benchmark-script "
            "(can be set multiple times)."
        ),
    )
    parser.add_argument(
        "--allow-empty-pack",
        action="store_true",
        help="Pass --allow-empty to the pack script.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining steps even if one command fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    python_exe = args.python_executable

    scripts_to_run: List[Path] = []
    if args.include_generate_data:
        scripts_to_run.append(Path("dev_scripts/benchmark_generate_data.py"))
    scripts_to_run.extend(BENCHMARK_SCRIPTS)

    missing_scripts = [
        script for script in scripts_to_run if not (project_root / script).exists()
    ]
    if missing_scripts:
        for script in missing_scripts:
            print(f"Missing script: {project_root / script}")
        return 1

    all_commands: List[List[str]] = []
    for script in scripts_to_run:
        all_commands.append([python_exe, str(script)])

    if not args.skip_pack:
        pack_cmd = [
            python_exe,
            "dev_scripts/pack_benchmark_jsons.py",
            "--project-root",
            str(project_root),
        ]
        if args.pack_output:
            pack_cmd.extend(["--output", str(args.pack_output.resolve())])
        if args.pack_benchmark_scripts:
            for benchmark_script in args.pack_benchmark_scripts:
                pack_cmd.extend(["--benchmark-script", benchmark_script])
        if args.allow_empty_pack:
            pack_cmd.append("--allow-empty")
        all_commands.append(pack_cmd)

    if args.dry_run:
        print("Dry run. Commands:")
        for cmd in all_commands:
            print(f"  {' '.join(cmd)}")
        return 0

    started_at = datetime.now()
    failures = 0

    for cmd in all_commands:
        return_code = _run_command(cmd=cmd, cwd=project_root)
        if return_code != 0:
            failures += 1
            if not args.continue_on_error:
                print("Stopping due to command failure.")
                break

    elapsed_total = datetime.now() - started_at
    print(f"\nTotal elapsed: {elapsed_total}")
    print(f"Failed commands: {failures}")

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
