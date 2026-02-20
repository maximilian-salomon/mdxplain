# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Benchmark profile script.
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

"""Run the Exact Memmap benchmark profile and persist benchmark metrics.

File Description
----------------
This script defines the "Exact Memmap" benchmark profile parameters and delegates
execution to the shared benchmark runner in ``benchmark_approx_memmap.py``.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/benchmark_exact_memmap.py``
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from benchmark_approx_memmap import _BenchmarkProfile, _parse_bool, _run_profile


results_dir = Path("benchmark_results_exact_memmap")
cache_root = Path("cache/benchmark_exact_memmap")
dataset_factors = [1, 2, 3, 5, 10]


def _exact_memmap_profile() -> _BenchmarkProfile:
    """Build Exact Memmap benchmark profile configuration.

    Parameters
    ----------
    None

    Returns
    -------
    _BenchmarkProfile
        Exact Memmap profile instance.

    Notes
    -----
    Exact Memmap profile uses memmap/chunking and disables Nyström decomposition.
    """
    # Build profile matching previous exact-memmap benchmark behavior.
    return _BenchmarkProfile(
        name="exact_memmap",
        results_dir=results_dir,
        cache_root=cache_root,
        dataset_factors=list(dataset_factors),
        use_memmap=True,
        chunk_size=2000,
        use_nystrom=False,
        n_landmarks=None,
        dpa_method="standard",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Exact Memmap benchmark execution.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.

    Notes
    -----
    Defaults preserve the existing full-profile behavior.
    """
    parser = argparse.ArgumentParser(description="Run Exact Memmap benchmark profile.")
    parser.add_argument(
        "--stacks",
        nargs="+",
        type=int,
        default=list(dataset_factors),
        help="Stack factors to run. Default: all configured factors.",
    )
    parser.add_argument("--remove", type=_parse_bool, default=True, help="Allow cleanup/overwrite behavior (true/false). Default: true.")
    return parser.parse_args()


def main() -> int:
    """Run the Exact Memmap benchmark profile.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Process-style exit code.

    Notes
    -----
    This CLI entry point delegates to the shared benchmark runner.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/benchmark_exact_memmap.py
    """
    # Build profile from CLI selection and execute via shared benchmark engine.
    args = parse_args()
    profile = replace(_exact_memmap_profile(), dataset_factors=list(args.stacks))
    return _run_profile(profile, remove=bool(args.remove))


if __name__ == "__main__":
    raise SystemExit(main())
