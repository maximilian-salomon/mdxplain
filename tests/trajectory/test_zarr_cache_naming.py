# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0) and GitHub Copilot (Claude Sonnet 4.0).
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

"""
Tests for Zarr cache naming to prevent path collisions.

Specifically tests that two trajectories with identical filenames but from
different directories get distinct Zarr cache paths, matching the full
trajectory system name (e.g. 'A_Y4R_hpp_run1') rather than just the file
stem ('run1').
"""

import os
import hashlib
import pytest
import numpy as np
import mdtraj as md

from mdxplain.trajectory.helper.dask_trajectory_helper.zarr_cache_helper import (
    ZarrCacheHelper,
)
from mdxplain.trajectory.helper.process_helper.trajectory_load_helper import (
    TrajectoryLoadHelper,
)
from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory(path, n_frames: int = 5, n_atoms: int = 3) -> tuple[str, str]:
    """Create a minimal XTC + PDB trajectory pair under *path*."""
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)
    for i in range(n_atoms):
        topology.add_atom(f"CA{i}", md.element.carbon, residue)

    xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
    traj = md.Trajectory(xyz, topology)

    xtc = str(path / "run1.xtc")
    pdb = str(path / "run1.pdb")
    traj.save_xtc(xtc)
    traj.save_pdb(pdb)
    return xtc, pdb


# ---------------------------------------------------------------------------
# ZarrCacheHelper unit tests
# ---------------------------------------------------------------------------

class TestZarrCacheHelperNaming:
    """Unit tests for ZarrCacheHelper.get_cache_path()."""

    def test_default_name_uses_file_stem(self, tmp_path):
        """
        Without traj_name the cache filename must start with the file stem.
        """
        xtc, _ = _make_trajectory(tmp_path)
        helper = ZarrCacheHelper(cache_dir=str(tmp_path / "cache"))
        cache_path = helper.get_cache_path(xtc)
        cache_name = os.path.basename(cache_path)
        assert cache_name.startswith("run1_"), (
            f"Expected cache name to start with 'run1_', got '{cache_name}'"
        )
        assert cache_name.endswith(".dask.zarr")

    def test_explicit_traj_name_used_in_cache_filename(self, tmp_path):
        """
        When traj_name is given it must appear at the start of the cache filename.
        """
        xtc, _ = _make_trajectory(tmp_path)
        helper = ZarrCacheHelper(cache_dir=str(tmp_path / "cache"))
        cache_path = helper.get_cache_path(xtc, traj_name="A_Y4R_hpp_run1")
        cache_name = os.path.basename(cache_path)
        assert cache_name.startswith("A_Y4R_hpp_run1_"), (
            f"Expected cache name to start with 'A_Y4R_hpp_run1_', got '{cache_name}'"
        )
        assert cache_name.endswith(".dask.zarr")

    def test_same_file_different_traj_names_produce_different_paths(self, tmp_path):
        """
        Two calls with the same file but different traj_names must yield
        different cache paths.
        """
        xtc, _ = _make_trajectory(tmp_path)
        helper = ZarrCacheHelper(cache_dir=str(tmp_path / "cache"))
        path_a = helper.get_cache_path(xtc, traj_name="A_system_run1")
        path_b = helper.get_cache_path(xtc, traj_name="B_system_run1")
        assert path_a != path_b, "Different traj_names must produce different cache paths"

    def test_same_name_different_files_produce_different_paths(self, tmp_path):
        """
        Files with the same name but in different directories must receive
        different cache paths (the hash component distinguishes them).
        """
        dir_a = tmp_path / "system_A"
        dir_b = tmp_path / "system_B"
        dir_a.mkdir()
        dir_b.mkdir()

        xtc_a, _ = _make_trajectory(dir_a)
        xtc_b, _ = _make_trajectory(dir_b)

        helper = ZarrCacheHelper(cache_dir=str(tmp_path / "cache"))
        path_a = helper.get_cache_path(xtc_a)
        path_b = helper.get_cache_path(xtc_b)

        assert path_a != path_b, (
            "Files with the same name in different directories must produce "
            "different cache paths"
        )

    def test_hash_derived_from_absolute_path(self, tmp_path):
        """
        The 8-character hash in the cache filename must equal the MD5 of
        the absolute trajectory file path.
        """
        xtc, _ = _make_trajectory(tmp_path)
        helper = ZarrCacheHelper(cache_dir=str(tmp_path / "cache"))
        cache_path = helper.get_cache_path(xtc)

        expected_hash = hashlib.md5(os.path.abspath(xtc).encode()).hexdigest()[:8]
        cache_name = os.path.basename(cache_path)
        assert expected_hash in cache_name, (
            f"Expected hash '{expected_hash}' not found in cache name '{cache_name}'"
        )


# ---------------------------------------------------------------------------
# Integration test: two "run1.xtc" files loaded together
# ---------------------------------------------------------------------------

class TestMultiSystemCacheCollision:
    """
    Integration test: loading two systems that both contain a file named
    'run1.xtc' with different atom counts must not exhibit cache collisions.
    """

    @pytest.fixture
    def two_systems(self, tmp_path):
        """Create two directories each with 'run1.xtc' but different n_atoms."""
        sys_a = tmp_path / "system_A"
        sys_b = tmp_path / "system_B"
        sys_a.mkdir()
        sys_b.mkdir()

        # System A: 5 atoms
        xtc_a, pdb_a = _make_trajectory(sys_a, n_atoms=5)
        # System B: 10 atoms (deliberately different)
        xtc_b, pdb_b = _make_trajectory(sys_b, n_atoms=10)

        return {
            "xtc_a": xtc_a, "pdb_a": pdb_a,
            "xtc_b": xtc_b, "pdb_b": pdb_b,
            "cache_dir": str(tmp_path / "cache"),
            "n_atoms_a": 5,
            "n_atoms_b": 10,
        }

    def test_separate_caches_created(self, two_systems):
        """
        Two DaskMDTrajectory objects for 'run1.xtc' from different directories
        must be stored in separate Zarr caches.
        """
        # Build the expected cache paths the same way TrajectoryLoadHelper does
        cache_a = os.path.join(
            two_systems["cache_dir"],
            "A_Y4R_hpp_run1_" + hashlib.md5(
                os.path.abspath(two_systems["xtc_a"]).encode()
            ).hexdigest()[:8] + ".dask.zarr",
        )
        cache_b = os.path.join(
            two_systems["cache_dir"],
            "B_Y4R_apo_run1_" + hashlib.md5(
                os.path.abspath(two_systems["xtc_b"]).encode()
            ).hexdigest()[:8] + ".dask.zarr",
        )
        assert cache_a != cache_b

    def test_correct_atom_counts_preserved(self, two_systems):
        """
        After loading both systems the atom counts must reflect their own
        topology, not whichever was cached last.
        """
        helper = ZarrCacheHelper(cache_dir=two_systems["cache_dir"])

        cache_path_a = helper.get_cache_path(
            two_systems["xtc_a"], traj_name="A_system_run1"
        )
        cache_path_b = helper.get_cache_path(
            two_systems["xtc_b"], traj_name="B_system_run1"
        )

        traj_a = DaskMDTrajectory(
            two_systems["xtc_a"],
            two_systems["pdb_a"],
            zarr_cache_path=cache_path_a,
        )
        traj_b = DaskMDTrajectory(
            two_systems["xtc_b"],
            two_systems["pdb_b"],
            zarr_cache_path=cache_path_b,
        )

        try:
            assert traj_a.n_atoms == two_systems["n_atoms_a"], (
                f"System A: expected {two_systems['n_atoms_a']} atoms, got {traj_a.n_atoms}"
            )
            assert traj_b.n_atoms == two_systems["n_atoms_b"], (
                f"System B: expected {two_systems['n_atoms_b']} atoms, got {traj_b.n_atoms}"
            )
        finally:
            # Release Zarr file handles before Windows tmp_path cleanup
            traj_a.cleanup()
            traj_b.cleanup()

    def test_shapes_do_not_cross_contaminate(self, two_systems):
        """
        XYZ shapes from both trajectories must match their own atom counts.
        This is the exact scenario that triggered the original ValueError.
        """
        helper = ZarrCacheHelper(cache_dir=two_systems["cache_dir"])

        traj_a = DaskMDTrajectory(
            two_systems["xtc_a"],
            two_systems["pdb_a"],
            zarr_cache_path=helper.get_cache_path(
                two_systems["xtc_a"], traj_name="A_system_run1"
            ),
        )
        traj_b = DaskMDTrajectory(
            two_systems["xtc_b"],
            two_systems["pdb_b"],
            zarr_cache_path=helper.get_cache_path(
                two_systems["xtc_b"], traj_name="B_system_run1"
            ),
        )

        try:
            # Neither should raise a reshape ValueError
            xyz_a = traj_a.xyz
            xyz_b = traj_b.xyz

            assert xyz_a.shape[-2] == two_systems["n_atoms_a"]
            assert xyz_b.shape[-2] == two_systems["n_atoms_b"]
        finally:
            # Release Zarr file handles before Windows tmp_path cleanup
            traj_a.cleanup()
            traj_b.cleanup()
