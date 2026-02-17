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

"""Run the Iterative benchmark profile and persist benchmark metrics.

File Description
----------------
This script defines the "Iterative" benchmark profile parameters and delegates
execution to the shared benchmark runner in ``benchmark_fast_standard.py``.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/benchmark_iterative.py``
"""

from __future__ import annotations

from pathlib import Path

from benchmark_fast_standard import _BenchmarkProfile, _run_profile


results_dir = Path("benchmark_results_iterative")
cache_root = Path("cache/benchmark_iterative")
dataset_factors = [1, 2, 3, 5, 10]


def _iterative_profile() -> _BenchmarkProfile:
    """Build Iterative benchmark profile configuration.

    Parameters
    ----------
    None

    Returns
    -------
    _BenchmarkProfile
        Iterative profile instance.

    Notes
    -----
    Iterative profile uses memmap/chunking and disables Nyström decomposition.
    """
    # Build profile matching previous iterative benchmark behavior.
    return _BenchmarkProfile(
        name="iterative",
        results_dir=results_dir,
        cache_root=cache_root,
        dataset_factors=list(dataset_factors),
        use_memmap=True,
        chunk_size=2000,
        use_nystrom=False,
        n_landmarks=None,
        dpa_method="knn_sampling",
    )


def main() -> int:
    """Run the Iterative benchmark profile.

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
    >>> # python dev_scripts/benchmark/benchmark_iterative.py
    """
    # Build profile and execute it via shared benchmark engine.
    return _run_profile(_iterative_profile())


if __name__ == "__main__":
    raise SystemExit(main())
