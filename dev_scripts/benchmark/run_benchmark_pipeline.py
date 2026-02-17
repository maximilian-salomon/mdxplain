# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# End-to-end benchmark pipeline orchestration script.
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

"""Run the complete benchmark pipeline with one command.

Pipeline stages:
1. Generate benchmark datasets.
2. Run benchmark profile scripts.
3. Package benchmark JSON outputs.
4. Generate benchmark analysis figures and CSV tables.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/run_benchmark_pipeline.py``
- ``python dev_scripts/benchmark/run_benchmark_pipeline.py --filetype svg``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


DATA_SCRIPT = Path("dev_scripts/benchmark/benchmark_generate_data.py")
PACK_SCRIPT = Path("dev_scripts/benchmark/pack_benchmark_jsons.py")
ANALYSIS_SCRIPT = Path("dev_scripts/benchmark/benchmark_analysis_report.py")
BENCHMARK_SCRIPTS = [
    Path("dev_scripts/benchmark/benchmark_approx_memmap.py"),
    Path("dev_scripts/benchmark/benchmark_exact_memmap.py"),
    Path("dev_scripts/benchmark/benchmark_exact_ram.py"),
]


@dataclass(frozen=True)
class PipelineCommand:
    """Container describing one executable benchmark pipeline step.

    Parameters
    ----------
    name : str
        Human-readable step name shown in console logs.
    command : list[str]
        Concrete command vector passed to ``subprocess.run``.

    Returns
    -------
    None
        Dataclass instances are consumed by the orchestration loop.

    Notes
    -----
    This dataclass keeps step metadata explicit and testable.
    """

    name: str
    command: list[str]


def _normalize_filetypes(filetypes: list[str]) -> list[str]:
    """Normalize and validate requested export file types.

    Parameters
    ----------
    filetypes : list[str]
        Raw file type values from CLI input.

    Returns
    -------
    list[str]
        Ordered unique list of normalized file extensions.

    Notes
    -----
    Allowed values are ``png`` and ``svg`` only.
    """
    # Keep only known extensions and preserve user-specified order.
    allowed = {"png", "svg"}
    normalized: list[str] = []
    for value in filetypes:
        ext = str(value).strip().lower().lstrip(".")
        if not ext:
            continue
        if ext not in allowed:
            raise ValueError(f"Unsupported file type {value!r}. Allowed: png, svg")
        if ext not in normalized:
            normalized.append(ext)
    return normalized or ["png"]


def _resolve_under_project(project_root: Path, path: Path) -> Path:
    """Resolve possibly-relative path under selected project root.

    Parameters
    ----------
    project_root : Path
        Root directory selected for benchmark execution.
    path : Path
        Candidate absolute or project-relative path.

    Returns
    -------
    Path
        Resolved absolute path.

    Notes
    -----
    Absolute input paths are preserved.
    """
    # Resolve relative paths against project root for deterministic behavior.
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _validate_required_scripts(project_root: Path, include_generate_data: bool) -> None:
    """Validate existence of scripts required by selected pipeline stages.

    Parameters
    ----------
    project_root : Path
        Root directory containing benchmark scripts.
    include_generate_data : bool
        Whether dataset generation step is enabled.

    Returns
    -------
    None
        Raises when a required script is missing.

    Notes
    -----
    This provides fail-fast diagnostics before any expensive benchmark work.
    """
    # Build script list for enabled stages and verify all files exist.
    required = [*BENCHMARK_SCRIPTS, PACK_SCRIPT, ANALYSIS_SCRIPT]
    if include_generate_data:
        required.insert(0, DATA_SCRIPT)

    missing = [path for path in required if not _resolve_under_project(project_root, path).exists()]
    if missing:
        details = "\n".join(f"- {_resolve_under_project(project_root, path)}" for path in missing)
        raise FileNotFoundError(f"Missing required benchmark scripts:\n{details}")


def _build_parser() -> argparse.ArgumentParser:
    """Create base argument parser for benchmark pipeline orchestration.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Parser instance with all benchmark pipeline options.

    Notes
    -----
    Parser construction is split into helper methods to keep functions short.
    """
    # Configure high-level parser metadata once.
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline and export analysis figures/tables.")

    # Add project and runtime execution controls.
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable for all sub-commands.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining steps after failures.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")

    # Add stage toggle options for generation, packaging, and analysis.
    parser.add_argument("--skip-generate-data", action="store_true", help="Skip dataset generation stage.")
    parser.add_argument("--skip-pack", action="store_true", help="Skip JSON packaging stage.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis figure export stage.")

    # Add output and analysis format options.
    parser.add_argument("--pack-output", type=Path, default=None, help="Optional output zip path for packaged JSON files.")
    parser.add_argument("--export-dir", type=Path, default=Path("benchmark/export"), help="Analysis figure/table export directory.")
    parser.add_argument("--filetype", action="append", default=None, choices=["png", "svg"], help="Analysis file type. Repeat to export both.")
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark pipeline execution.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed benchmark pipeline options.

    Notes
    -----
    ``--filetype`` defaults to ``png`` when omitted.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/run_benchmark_pipeline.py --dry-run
    """
    # Build parser and parse CLI values once.
    return _build_parser().parse_args()


def _append_data_step(commands: list[PipelineCommand], python_exe: str, include_generate_data: bool) -> None:
    """Append dataset generation step when enabled.

    Parameters
    ----------
    commands : list[PipelineCommand]
        Mutable command list being assembled.
    python_exe : str
        Python executable path for subprocess execution.
    include_generate_data : bool
        Whether dataset generation stage is enabled.

    Returns
    -------
    None
        Commands list is updated in place.

    Notes
    -----
    Generation stage is skipped when ``include_generate_data`` is false.
    """
    # Add generation command only when requested by current flow settings.
    if include_generate_data:
        commands.append(PipelineCommand("Generate benchmark datasets", [python_exe, str(DATA_SCRIPT)]))


def _append_profile_steps(commands: list[PipelineCommand], python_exe: str) -> None:
    """Append benchmark profile script execution steps.

    Parameters
    ----------
    commands : list[PipelineCommand]
        Mutable command list being assembled.
    python_exe : str
        Python executable path for subprocess execution.

    Returns
    -------
    None
        Commands list is updated in place.

    Notes
    -----
    Profile scripts are added in deterministic fixed order.
    """
    # Add one execution step per benchmark profile script.
    for script in BENCHMARK_SCRIPTS:
        commands.append(PipelineCommand(f"Run {script.stem}", [python_exe, str(script)]))


def _append_pack_step(
    commands: list[PipelineCommand],
    python_exe: str,
    include_pack: bool,
    project_root: Path,
    pack_output: Path | None,
) -> None:
    """Append packaging step when enabled.

    Parameters
    ----------
    commands : list[PipelineCommand]
        Mutable command list being assembled.
    python_exe : str
        Python executable path for subprocess execution.
    include_pack : bool
        Whether JSON packaging stage is enabled.
    project_root : Path
        Project root used for absolute path resolution.
    pack_output : Path or None
        Optional custom archive output path.

    Returns
    -------
    None
        Commands list is updated in place.

    Notes
    -----
    Packaging command always receives explicit ``--project-root``.
    """
    # Skip pack stage entirely when disabled.
    if not include_pack:
        return

    # Build base packaging command and optional custom output flag.
    command = [python_exe, str(PACK_SCRIPT), "--project-root", str(project_root)]
    if pack_output is not None:
        command.extend(["--output", str(_resolve_under_project(project_root, pack_output))])

    commands.append(PipelineCommand("Package benchmark JSON outputs", command))


def _append_analysis_step(
    commands: list[PipelineCommand],
    python_exe: str,
    include_analysis: bool,
    project_root: Path,
    export_dir: Path,
    filetypes: list[str],
) -> None:
    """Append analysis export step when enabled.

    Parameters
    ----------
    commands : list[PipelineCommand]
        Mutable command list being assembled.
    python_exe : str
        Python executable path for subprocess execution.
    include_analysis : bool
        Whether analysis export stage is enabled.
    project_root : Path
        Benchmark root passed to analysis script.
    export_dir : Path
        Target directory for figure/table exports.
    filetypes : list[str]
        Normalized list of requested export file types.

    Returns
    -------
    None
        Commands list is updated in place.

    Notes
    -----
    File types are forwarded as repeated ``--filetype`` flags.
    """
    # Skip analysis stage entirely when disabled.
    if not include_analysis:
        return

    # Build analysis command with explicit benchmark root and export directory.
    export_root = _resolve_under_project(project_root, export_dir)
    command = [python_exe, str(ANALYSIS_SCRIPT), "--benchmark-root", str(project_root), "--export-dir", str(export_root)]
    for ext in filetypes:
        command.extend(["--filetype", ext])

    commands.append(PipelineCommand("Generate benchmark analysis figures and tables", command))


def _build_pipeline_commands(
    project_root: Path,
    python_exe: str,
    include_generate_data: bool,
    include_pack: bool,
    include_analysis: bool,
    pack_output: Path | None,
    export_dir: Path,
    filetypes: list[str],
) -> list[PipelineCommand]:
    """Build full ordered benchmark command sequence.

    Parameters
    ----------
    project_root : Path
        Project root for path resolution.
    python_exe : str
        Python executable used for subprocess commands.
    include_generate_data : bool
        Toggle for dataset generation stage.
    include_pack : bool
        Toggle for JSON packaging stage.
    include_analysis : bool
        Toggle for analysis export stage.
    pack_output : Path or None
        Optional custom output zip path.
    export_dir : Path
        Analysis figure export directory.
    filetypes : list[str]
        Requested analysis export file types.

    Returns
    -------
    list[PipelineCommand]
        Ordered list of commands to execute.

    Notes
    -----
    Commands are produced without side effects.
    """
    # Assemble pipeline commands in canonical execution order.
    commands: list[PipelineCommand] = []
    _append_data_step(commands, python_exe, include_generate_data)
    _append_profile_steps(commands, python_exe)
    _append_pack_step(commands, python_exe, include_pack, project_root, pack_output)
    _append_analysis_step(commands, python_exe, include_analysis, project_root, export_dir, filetypes)
    return commands


def _run_command(step: PipelineCommand, cwd: Path, dry_run: bool) -> int:
    """Run one benchmark pipeline command and return process exit code.

    Parameters
    ----------
    step : PipelineCommand
        Pipeline step metadata and executable command vector.
    cwd : Path
        Working directory used for subprocess execution.
    dry_run : bool
        When true, prints command and skips actual execution.

    Returns
    -------
    int
        Exit code from command execution or ``0`` in dry-run mode.

    Notes
    -----
    Runtime duration is always reported for executed commands.
    """
    # Print stable command trace for human-readable pipeline logs.
    print(f"\n=== {step.name}")
    print(f"$ {' '.join(step.command)}")
    if dry_run:
        return 0

    # Execute subprocess and report elapsed runtime with exit code.
    started = time.perf_counter()
    completed = subprocess.run(step.command, cwd=str(cwd), check=False)
    print(f"--- exit={completed.returncode} elapsed={time.perf_counter() - started:.1f}s")
    return int(completed.returncode)


def _execute_pipeline(commands: list[PipelineCommand], project_root: Path, continue_on_error: bool, dry_run: bool) -> int:
    """Execute prepared benchmark pipeline commands sequentially.

    Parameters
    ----------
    commands : list[PipelineCommand]
        Ordered command list produced by command builder.
    project_root : Path
        Working directory for subprocess execution.
    continue_on_error : bool
        Whether execution continues after a failing step.
    dry_run : bool
        Whether commands are printed only.

    Returns
    -------
    int
        Number of failed commands.

    Notes
    -----
    Execution stops at first failure unless ``continue_on_error`` is enabled.
    """
    # Run all steps sequentially and stop early only when requested.
    failures = 0
    for step in commands:
        code = _run_command(step, project_root, dry_run)
        if code == 0:
            continue
        failures += 1
        if not continue_on_error:
            print("Stopping after first failure. Use --continue-on-error to continue.")
            break
    return failures


def main() -> int:
    """Run configured benchmark pipeline stages and return process exit code.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Exit code ``0`` when no command failed, otherwise ``1``.

    Notes
    -----
    Missing required scripts are reported before command execution starts.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/run_benchmark_pipeline.py --dry-run
    """
    # Parse arguments and normalize commonly reused runtime settings.
    args = parse_args()
    project_root = args.project_root.resolve()
    filetypes = _normalize_filetypes(args.filetype or [])

    # Validate required scripts and assemble command execution plan.
    _validate_required_scripts(project_root, include_generate_data=not args.skip_generate_data)
    commands = _build_pipeline_commands(
        project_root,
        args.python_executable,
        not args.skip_generate_data,
        not args.skip_pack,
        not args.skip_analysis,
        args.pack_output,
        args.export_dir,
        filetypes,
    )

    # Execute pipeline and report final failure summary with elapsed time.
    started_at = datetime.now()
    failures = _execute_pipeline(commands, project_root, args.continue_on_error, args.dry_run)
    print(f"\nTotal elapsed: {datetime.now() - started_at}")
    print(f"Failed steps: {failures}")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
