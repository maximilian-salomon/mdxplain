# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Legacy benchmark compatibility runner script.
#
# Author: Maximilian Salomon
# Created with assistance from GPT-5.3-Codex.
#
# Copyright (C) 2025 Maximilian Salomon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Compatibility entrypoint for benchmark execution and JSON packaging.

This script preserves the historical CLI while delegating execution to
``run_benchmark_pipeline.py``.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/run_all_benchmarks_and_pack.py``
- ``python dev_scripts/benchmark/run_all_benchmarks_and_pack.py --include-generate-data``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PIPELINE_SCRIPT = Path("dev_scripts/benchmark/run_benchmark_pipeline.py")


def parse_args() -> argparse.Namespace:
    """Parse legacy CLI arguments for compatibility benchmark runner.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed legacy options mapped to the new pipeline runner.

    Notes
    -----
    ``--allow-empty-pack`` and ``--pack-benchmark-script`` are accepted for
    compatibility and ignored by the delegated pipeline runner.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/run_all_benchmarks_and_pack.py --dry-run
    """
    # Keep legacy options stable while internally delegating behavior.
    parser = argparse.ArgumentParser(description="Compatibility wrapper for benchmark run + pack flow.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable for delegated commands.")
    parser.add_argument("--include-generate-data", action="store_true", help="Include data generation stage.")
    parser.add_argument("--skip-pack", action="store_true", help="Skip packaging stage.")
    parser.add_argument("--pack-output", type=Path, default=None, help="Optional output zip for packaged JSON files.")
    parser.add_argument("--allow-empty-pack", action="store_true", help="Accepted for compatibility; no effect.")
    parser.add_argument("--pack-benchmark-script", action="append", dest="pack_benchmark_scripts", help="Accepted for compatibility; no effect.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining steps after failures.")
    parser.add_argument("--dry-run", action="store_true", help="Print delegated command only.")
    return parser.parse_args()


def _build_pipeline_command(args: argparse.Namespace) -> list[str]:
    """Build delegated command invoking the new benchmark pipeline runner.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed legacy compatibility options.

    Returns
    -------
    list[str]
        Delegated command vector for ``run_benchmark_pipeline.py``.

    Notes
    -----
    Defaults match legacy behavior: generation disabled unless explicitly
    requested by ``--include-generate-data``.
    """
    # Build baseline command with explicit project and interpreter settings.
    command = [
        args.python_executable,
        str(PIPELINE_SCRIPT),
        "--project-root",
        str(args.project_root.resolve()),
        "--python-executable",
        args.python_executable,
    ]

    # Preserve legacy default by skipping generation unless explicitly requested.
    if not args.include_generate_data:
        command.append("--skip-generate-data")
    if args.skip_pack:
        command.append("--skip-pack")
    if args.pack_output is not None:
        command.extend(["--pack-output", str(args.pack_output.resolve())])
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.dry_run:
        command.append("--dry-run")

    return command


def main() -> int:
    """Run delegated benchmark pipeline command and return subprocess exit code.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Exit code from delegated pipeline execution.

    Notes
    -----
    Delegation keeps this script thin and backward compatible.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/run_all_benchmarks_and_pack.py --dry-run
    """
    # Parse legacy arguments and translate to pipeline command.
    args = parse_args()
    command = _build_pipeline_command(args)

    # Execute delegated command in selected project root.
    completed = subprocess.run(command, cwd=str(args.project_root.resolve()), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
