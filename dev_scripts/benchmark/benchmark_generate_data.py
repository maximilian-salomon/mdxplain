# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Benchmark dataset generation script.
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

"""Generate stacked benchmark trajectories from the 2RJY source dataset.

The script creates stacked trajectory variants with reproducible Gaussian noise
and stores them under ``data/benchmarks``.

How To Use
----------
Run from project root:

- ``python dev_scripts/benchmark/benchmark_generate_data.py``
- ``python dev_scripts/benchmark/benchmark_generate_data.py --factors 2 3 5 10 30 50``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import mdtraj as md
import numpy as np


def _find_input_files(data_dir: Path) -> tuple[Path, Path]:
    """Find one topology file and one trajectory file in source data directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing source topology and trajectory files.

    Returns
    -------
    tuple[Path, Path]
        Tuple containing ``(topology_pdb, trajectory_file)``.

    Notes
    -----
    Topology lookup is restricted to ``.pdb`` files. Trajectory lookup accepts
    common MD trajectory extensions used in this project.
    """
    # Define accepted trajectory extensions for source dataset discovery.
    traj_exts = {".xtc", ".dcd", ".trr", ".nc", ".mdcrd", ".h5", ".dtr"}

    # Resolve first topology and trajectory files in deterministic order.
    pdb_files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".pdb")
    traj_files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in traj_exts)

    # Fail early with explicit messages when required source files are missing.
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb topology found in {data_dir}")
    if not traj_files:
        raise FileNotFoundError(f"No trajectory file found in {data_dir}")

    return pdb_files[0], traj_files[0]


def _build_stacked_trajectory(
    src_traj: Path,
    src_pdb: Path,
    factor: int,
    noise_sigma_nm: float,
    seed: int,
) -> md.Trajectory:
    """Load, stack, and optionally perturb the source trajectory.

    Parameters
    ----------
    src_traj : Path
        Input trajectory file path.
    src_pdb : Path
        Input topology path.
    factor : int
        Number of times frames are repeated along the frame axis.
    noise_sigma_nm : float
        Standard deviation of Gaussian coordinate noise in nanometers.
    seed : int
        Base random seed used to create deterministic noise per factor.

    Returns
    -------
    md.Trajectory
        New trajectory instance containing stacked frames.

    Notes
    -----
    Noise is generated with ``numpy.random.default_rng`` and cast to the source
    coordinate dtype to avoid unnecessary precision changes.
    """
    # Load source trajectory with explicit topology to preserve atom mapping.
    source = md.load(str(src_traj), top=str(src_pdb))

    # Stack coordinates and timestamps by requested multiplicative factor.
    stacked_xyz = np.concatenate([source.xyz] * factor, axis=0)
    stacked_time = np.concatenate([source.time] * factor, axis=0)

    # Apply optional reproducible Gaussian perturbation to stacked coordinates.
    if noise_sigma_nm > 0.0:
        rng = np.random.default_rng(seed + factor)
        noise = rng.normal(0.0, noise_sigma_nm, size=stacked_xyz.shape)
        stacked_xyz = stacked_xyz + noise.astype(stacked_xyz.dtype, copy=False)

    return md.Trajectory(xyz=stacked_xyz, topology=source.topology, time=stacked_time)


def _write_stacked_dataset(
    src_pdb: Path,
    src_traj: Path,
    out_dir: Path,
    factor: int,
    noise_sigma_nm: float,
    seed: int,
) -> None:
    """Write one stacked benchmark dataset directory.

    Parameters
    ----------
    src_pdb : Path
        Source topology file.
    src_traj : Path
        Source trajectory file.
    out_dir : Path
        Target directory for generated files.
    factor : int
        Stacking factor used to repeat trajectory frames.
    noise_sigma_nm : float
        Standard deviation of Gaussian coordinate noise in nanometers.
    seed : int
        Base random seed for deterministic coordinate noise.

    Returns
    -------
    None
        Files are written to disk under ``out_dir``.

    Notes
    -----
    The topology file is copied unchanged. The trajectory output is always
    written in XTC format with the original trajectory stem.
    """
    # Ensure destination directory exists before writing files.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy topology and write generated stacked trajectory to destination.
    out_pdb = out_dir / src_pdb.name
    out_traj = out_dir / src_traj.name
    shutil.copy2(src_pdb, out_pdb)

    # Build and persist stacked trajectory for this benchmark factor.
    stacked = _build_stacked_trajectory(src_traj, src_pdb, factor, noise_sigma_nm, seed)
    stacked.save_xtc(str(out_traj))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark dataset generation.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed command-line configuration for generation settings.

    Notes
    -----
    Default values are aligned with the existing benchmark workflow.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/benchmark_generate_data.py --noise-sigma-nm 0.02
    """
    parser = argparse.ArgumentParser(description="Generate stacked benchmark datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/2RJY"), help="Source data directory.")
    parser.add_argument("--out-root", type=Path, default=Path("data/benchmarks"), help="Output root directory.")
    parser.add_argument("--noise-sigma-nm", type=float, default=0.01, help="Gaussian noise sigma in nm.")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed for noise generation.")
    parser.add_argument(
        "--factors",
        nargs="+",
        type=int,
        default=[2, 3, 5, 10, 30, 50],
        help="Stacking factors to generate.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate all configured stacked benchmark datasets.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Exit code ``0`` on success.

    Notes
    -----
    The source dataset is read once and transformed separately for each factor.

    Examples
    --------
    >>> # CLI usage
    >>> # python dev_scripts/benchmark/benchmark_generate_data.py
    """
    # Parse generation settings from CLI and resolve source files.
    args = parse_args()
    src_pdb, src_traj = _find_input_files(args.data_dir)

    # Generate one benchmark dataset directory per requested factor.
    for factor in args.factors:
        name = f"2RJY_stack{factor}x"
        out_dir = args.out_root / name
        _write_stacked_dataset(src_pdb, src_traj, out_dir, factor, args.noise_sigma_nm, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
