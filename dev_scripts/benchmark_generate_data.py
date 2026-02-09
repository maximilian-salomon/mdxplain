#!/usr/bin/env python
# Generate stacked benchmark trajectories with Gaussian noise.

from __future__ import annotations

from pathlib import Path
import shutil

import mdtraj as md
import numpy as np


def _find_input_files(data_dir: Path) -> tuple[Path, Path]:
    traj_exts = {".xtc", ".dcd", ".trr", ".nc", ".mdcrd", ".h5", ".dtr"}
    pdb_files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".pdb")
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb topology found in {data_dir}")
    traj_files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in traj_exts)
    if not traj_files:
        raise FileNotFoundError(f"No trajectory file found in {data_dir}")
    return pdb_files[0], traj_files[0]


def _write_stacked_trajectory(
    src_pdb: Path,
    src_traj: Path,
    out_dir: Path,
    factor: int,
    noise_sigma_nm: float,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = out_dir / src_pdb.name
    out_traj = out_dir / src_traj.name

    shutil.copy2(src_pdb, out_pdb)

    rng = np.random.default_rng(seed + factor)
    traj = md.load(str(src_traj), top=str(src_pdb))
    coords = traj.xyz
    stacked = np.concatenate([coords] * factor, axis=0)
    if noise_sigma_nm > 0:
        noise = rng.normal(0.0, noise_sigma_nm, size=stacked.shape).astype(
            stacked.dtype, copy=False
        )
        stacked = stacked + noise

    stacked_time = np.concatenate([traj.time] * factor, axis=0)
    stacked_traj = md.Trajectory(
        xyz=stacked,
        topology=traj.topology,
        time=stacked_time,
    )
    stacked_traj.save_xtc(str(out_traj))


def main() -> int:
    data_dir = Path("data/2RJY")
    out_root = Path("data/benchmarks")

    noise_sigma_nm = 0.01
    seed = 7
    factors = [2, 3, 5, 10, 30, 50]

    src_pdb, src_traj = _find_input_files(data_dir)

    for factor in factors:
        name = f"2RJY_stack{factor}x"
        out_dir = out_root / name
        _write_stacked_trajectory(
            src_pdb=src_pdb,
            src_traj=src_traj,
            out_dir=out_dir,
            factor=factor,
            noise_sigma_nm=noise_sigma_nm,
            seed=seed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
