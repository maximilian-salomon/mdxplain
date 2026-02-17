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

"""Run the Standard Full benchmark profile and persist benchmark metrics.

File Description
----------------
This script defines the "Standard Full" benchmark profile parameters and
reuses the shared benchmark runner in ``benchmark_fast_standard.py``.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/benchmark_standard_full.py``
"""

from __future__ import annotations

from pathlib import Path

from benchmark_fast_standard import _BenchmarkProfile, _run_profile


results_dir = Path("benchmark_results_standard_full")
cache_root = Path("cache/benchmark_standard_full")
dataset_factors = [1, 2, 3, 5]


def _standard_full_profile() -> _BenchmarkProfile:
    """Build Standard Full benchmark profile configuration.

    Parameters
    ----------
    None

    Returns
    -------
    _BenchmarkProfile
        Standard Full profile instance.

    Notes
    -----
    Standard Full profile disables memmap/chunking and uses standard DPA.
    """
    # Build profile matching previous standard-full benchmark behavior.
    return _BenchmarkProfile(
        name="standard_full",
        results_dir=results_dir,
        cache_root=cache_root,
        dataset_factors=list(dataset_factors),
        use_memmap=False,
        chunk_size=None,
        use_nystrom=False,
        n_landmarks=None,
        dpa_method="standard",
    )


def main() -> int:
    """Run the Standard Full benchmark profile.

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
    >>> # python dev_scripts/benchmark/benchmark_standard_full.py
    """
    # Build profile and execute it via shared benchmark engine.
    return _run_profile(_standard_full_profile())


if __name__ == "__main__":
    raise SystemExit(main())
