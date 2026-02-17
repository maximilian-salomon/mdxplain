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

"""Run the Exact RAM benchmark profile and persist benchmark metrics.

File Description
----------------
This script defines the "Exact RAM" benchmark profile parameters and
reuses the shared benchmark runner in ``benchmark_approx_memmap.py``.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/benchmark_exact_ram.py``
"""

from __future__ import annotations

from pathlib import Path

from benchmark_approx_memmap import _BenchmarkProfile, _run_profile


results_dir = Path("benchmark_results_exact_ram")
cache_root = Path("cache/benchmark_exact_ram")
dataset_factors = [1, 2, 3, 5]


def _exact_ram_profile() -> _BenchmarkProfile:
    """Build Exact RAM benchmark profile configuration.

    Parameters
    ----------
    None

    Returns
    -------
    _BenchmarkProfile
        Exact RAM profile instance.

    Notes
    -----
    Exact RAM profile disables memmap/chunking and uses standard DPA.
    """
    # Build profile matching previous exact-ram benchmark behavior.
    return _BenchmarkProfile(
        name="exact_ram",
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
    """Run the Exact RAM benchmark profile.

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
    >>> # python dev_scripts/benchmark/benchmark_exact_ram.py
    """
    # Build profile and execute it via shared benchmark engine.
    return _run_profile(_exact_ram_profile())


if __name__ == "__main__":
    raise SystemExit(main())
