#!/usr/bin/env python
# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from OpenAI Codex (GPT-5.3-Codex High).
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Create a shareable pipeline archive from trajectory data in ``Data/...``.

Usage
-----
Run from project root:

``python3 tests/manual_tests/create_pipeline_archive_from_data.py``

Examples
--------
Minimal (uses repo default ``Data/2RJY``):

``python3 tests/manual_tests/create_pipeline_archive_from_data.py``

With explicit input path and archive output:

``python3 tests/manual_tests/create_pipeline_archive_from_data.py Data/2RJY --output-archive pipeline_from_data.tar.zst --compression zst --use-memmap``

Default output archive is created in the project root:

``pipeline_from_data.tar.zst``

This is the same default archive path used by:

``tests/manual_tests/validate_pipeline_archive_cross_platform.py``

Result markers:
- ``TEST RESULT: PASS`` and ``TEST_RESULT=PASS`` for success.
- ``TEST RESULT: FAIL`` and ``TEST_RESULT=FAIL`` for failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, TYPE_CHECKING

# Allow direct script execution from the repository without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from mdxplain.pipeline.manager.pipeline_manager import PipelineManager

DEFAULT_DATA_INPUT = Path("Data/2RJY")
DEFAULT_OUTPUT_ARCHIVE = Path("pipeline_from_data.tar.zst")


def _print_test_result(success: bool) -> None:
    """Print a clear and machine-readable final result line.

    Parameters
    ----------
    success : bool
        Final script status.
    """
    label = "PASS" if success else "FAIL"
    print(f"TEST RESULT: {label}")
    print(f"TEST_RESULT={label}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    examples = (
        "Examples:\n"
        "  python3 tests/manual_tests/create_pipeline_archive_from_data.py\n"
        "  python3 tests/manual_tests/create_pipeline_archive_from_data.py "
        "Data/2RJY --output-archive pipeline_from_data.tar.zst\n"
        "  python3 tests/manual_tests/create_pipeline_archive_from_data.py "
        "Data/2RJY --no-use-memmap --compression gz --stride 5\n"
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Load trajectories from Data/... into PipelineManager and create "
            "a shareable archive."
        ),
        epilog=examples,
    )
    parser.add_argument(
        "data_input",
        type=Path,
        nargs="?",
        default=DEFAULT_DATA_INPUT,
        help=(
            "Path to trajectory directory. "
            f"Default: {DEFAULT_DATA_INPUT}"
        ),
    )
    parser.add_argument(
        "--output-archive",
        type=Path,
        default=DEFAULT_OUTPUT_ARCHIVE,
        help=(
            "Target archive path (extension can be included). "
            f"Default: {DEFAULT_OUTPUT_ARCHIVE}"
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("zst", "gz", "bz2"),
        default="zst",
        help="Archive compression method.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("benchmark_results/archive_build/cache"),
        help="Base cache directory used by PipelineManager.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Trajectory loading stride (load every n-th frame).",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Concatenate multiple trajectories per system.",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default=None,
        help="Optional MDTraj atom selection applied during load.",
    )
    parser.add_argument(
        "--use-memmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable memmap-backed pipeline data.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size for memmap-backed processing.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show tqdm progress bars.",
    )
    parser.add_argument(
        "--add-labels",
        action="store_true",
        help="Add default non-consensus labels to all loaded trajectories.",
    )
    parser.add_argument(
        "--include-visualizations",
        action="store_true",
        help="Include visualization outputs in the archive.",
    )
    parser.add_argument(
        "--no-include-structure-files",
        action="store_true",
        help="Exclude structure files (PDB/PML) from archive.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="Optional compression level override.",
    )
    return parser.parse_args()


def _print_startup_explanation(args: argparse.Namespace, data_input: Path) -> None:
    """Print startup explanation and effective settings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    data_input : pathlib.Path
        Resolved trajectory data input path.
    """
    print("=== Create Pipeline Archive From Data ===")
    print(
        "Goal: load trajectories from Data/... into a fresh pipeline and "
        "export one shareable archive."
    )
    print(
        "Input: "
        f"data_input={data_input}, "
        f"compression={args.compression}, use_memmap={args.use_memmap}, "
        f"output_archive={Path(args.output_archive).resolve()}"
    )
    print(f"Default dataset path: {DEFAULT_DATA_INPUT}")
    print(f"Default archive path: {DEFAULT_OUTPUT_ARCHIVE}")
    print(
        "How to run: python3 "
        "tests/manual_tests/create_pipeline_archive_from_data.py "
        "[Data/<dataset_dir>]"
    )
    print(
        "Help: python3 tests/manual_tests/create_pipeline_archive_from_data.py --help"
    )
    print(
        "Validate next: python3 "
        "tests/manual_tests/validate_pipeline_archive_cross_platform.py"
    )


def _validate_inputs(data_input: Path, stride: int) -> bool:
    """Validate user-provided input path and basic options.

    Parameters
    ----------
    data_input : pathlib.Path
        Data input path expected by ``pipeline.trajectory.load_trajectories``.
    stride : int
        Requested trajectory loading stride.

    Returns
    -------
    bool
        ``True`` when inputs are valid, otherwise ``False``.
    """
    if not data_input.exists():
        print(f"Data path not found: {data_input}")
        return False
    if stride < 1:
        print("--stride must be >= 1")
        return False
    return True


def _count_loaded_trajectories(pipeline: "PipelineManager") -> int:
    """Return number of loaded trajectories from pipeline internals.

    Parameters
    ----------
    pipeline : PipelineManager
        Active pipeline instance.

    Returns
    -------
    int
        Number of loaded trajectory objects.
    """
    trajectories = getattr(pipeline._data.trajectory_data, "trajectories", [])
    return len(trajectories)


def _create_archive(args: argparse.Namespace, data_input: Path) -> Path:
    """Create an archive from loaded trajectories.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    data_input : pathlib.Path
        Resolved trajectory input path.

    Returns
    -------
    pathlib.Path
        Absolute path to the created archive.

    Raises
    ------
    RuntimeError
        Raised when no trajectories were loaded.
    """
    from mdxplain.pipeline.manager.pipeline_manager import PipelineManager

    cache_dir = args.cache_dir.resolve()
    output_archive = args.output_archive.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_archive.parent.mkdir(parents=True, exist_ok=True)

    pipeline: Optional[PipelineManager] = None
    try:
        # Create pipeline with explicit settings so archives are reproducible.
        pipeline = PipelineManager(
            stride=args.stride,
            concat=args.concat,
            selection=args.selection,
            use_memmap=args.use_memmap,
            chunk_size=args.chunk_size,
            cache_dir=str(cache_dir),
            show_progress=args.show_progress,
        )

        # Step 1: Load trajectories from Data/... (directory or file list path).
        pipeline.trajectory.load_trajectories(str(data_input))
        trajectory_count = _count_loaded_trajectories(pipeline)
        if trajectory_count < 1:
            raise RuntimeError("No trajectories were loaded from the provided data path.")
        print(f"Loaded trajectories: {trajectory_count}")

        # Step 2: Optional label enrichment for convenience.
        if args.add_labels:
            pipeline.trajectory.add_labels(traj_selection="all")
            print("Added default labels to all trajectories.")

        # Step 3: Create archive compatible with validate script.
        created_archive = Path(
            pipeline.create_sharable_archive(
                str(output_archive),
                compression=args.compression,
                exclude_visualizations=not args.include_visualizations,
                include_structure_files=not args.no_include_structure_files,
                compression_level=args.compression_level,
            )
        ).resolve()
        return created_archive
    finally:
        if pipeline is not None:
            pipeline.close()


def main() -> int:
    """Run archive creation workflow from trajectory data input.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on failure).
    """
    # Phase 1: Parse and validate input.
    args = _parse_args()
    data_input = args.data_input.resolve()
    if not _validate_inputs(data_input=data_input, stride=args.stride):
        _print_test_result(False)
        return 1

    # Phase 2: Print startup explanation for manual runs.
    _print_startup_explanation(args, data_input)

    # Phase 3: Execute archive creation and report result.
    try:
        archive_path = _create_archive(args, data_input)
        print("=== Archive Creation Summary ===")
        print(f"Data input: {data_input}")
        print(f"Archive created: {archive_path}")
        _print_test_result(True)
        return 0
    except Exception as exc:
        print(f"Archive creation failed: {type(exc).__name__}: {exc}")
        _print_test_result(False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
