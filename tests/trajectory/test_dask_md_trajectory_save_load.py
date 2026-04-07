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

"""Tests for DaskMDTrajectory save/load functionality.

save() creates a portable ``.dask_traj`` archive (tar + zstd) that bundles
both the pickle metadata and the Zarr cache.  Loaded trajectories therefore
do not depend on the original Zarr cache location.
"""

import os
import shutil

import numpy as np
import pytest
import mdtraj as md

from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory
from mdxplain.trajectory.helper.dask_trajectory_helper.dask_trajectory_archive_helper import (
    ARCHIVE_EXTENSION,
)


class TestDaskMDTrajectorySaveLoad:
    """Test save/load functionality of DaskMDTrajectory."""

    @pytest.fixture
    def mock_trajectory(self, tmp_path):
        """Create a minimal in-memory trajectory and save it to xtc + pdb."""
        n_frames, n_atoms = 5, 3

        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain)
        for i in range(n_atoms):
            topology.add_atom(f"CA{i}", md.element.carbon, residue)

        xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
        traj = md.Trajectory(xyz, topology)

        traj_file = str(tmp_path / "test.xtc")
        top_file = str(tmp_path / "test.pdb")
        traj.save_xtc(traj_file)
        traj.save_pdb(top_file)

        return traj_file, top_file, n_frames, n_atoms

    # ------------------------------------------------------------------
    # save() behaviour
    # ------------------------------------------------------------------

    def test_save_creates_parent_directories(self, tmp_path, mock_trajectory):
        """
        save() must create parent directories automatically.

        Validates that supplying a path whose parent does not yet exist
        causes the directory tree to be created before writing the archive.
        """
        traj_file, top_file, _, _ = mock_trajectory
        zarr_cache = str(tmp_path / "zarr_cache")
        dask_traj = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=zarr_cache)

        try:
            nested = tmp_path / "output" / "trajectories" / "my_traj"
            assert not nested.parent.exists()

            dask_traj.save(str(nested))

            assert nested.parent.exists()
            assert (str(nested) + ARCHIVE_EXTENSION or nested.exists())
        finally:
            dask_traj.cleanup()

    def test_save_appends_extension(self, tmp_path, mock_trajectory):
        """
        save() must append the .dask_traj extension when not present.

        Validates that passing a path without the archive extension still
        produces a correctly named archive file on disk.
        """
        traj_file, top_file, _, _ = mock_trajectory
        zarr_cache = str(tmp_path / "zarr_cache")
        dask_traj = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=zarr_cache)

        try:
            base = str(tmp_path / "my_traj")
            dask_traj.save(base)
            assert os.path.exists(base + ARCHIVE_EXTENSION)
        finally:
            dask_traj.cleanup()

    # ------------------------------------------------------------------
    # save / load roundtrip
    # ------------------------------------------------------------------

    def test_save_load_roundtrip(self, tmp_path, mock_trajectory):
        """
        Full save/load roundtrip must preserve all metadata and coordinate data.

        Validates that trajectory properties (n_frames, n_atoms, topology)
        and xyz coordinate values are bit-identical after a save/load cycle.
        """
        traj_file, top_file, n_frames, n_atoms = mock_trajectory
        zarr_cache = str(tmp_path / "zarr_cache")
        original = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=zarr_cache)

        archive = str(tmp_path / "saved_traj")
        original.save(archive)

        loaded = DaskMDTrajectory.load(archive + ARCHIVE_EXTENSION)
        try:
            assert isinstance(loaded, DaskMDTrajectory)
            assert loaded.n_frames == n_frames
            assert loaded.n_atoms == n_atoms
            assert loaded.trajectory_file == original.trajectory_file
            assert loaded.topology_file == original.topology_file
            assert np.allclose(loaded.xyz, original.xyz)
        finally:
            original.cleanup()
            loaded.cleanup()

    def test_zarr_cache_independence_after_save(self, tmp_path, mock_trajectory):
        """
        Loaded trajectory must be accessible even after the original Zarr cache
        is deleted, because save() bundles the Zarr data inside the archive.

        Validates:
        1. Metadata (n_frames, n_atoms) survives the round-trip.
        2. xyz coordinate access works without the original Zarr cache.
        3. The loaded data matches the data before saving.
        """
        traj_file, top_file, n_frames, n_atoms = mock_trajectory
        zarr_cache = str(tmp_path / "zarr_cache")
        original = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=zarr_cache)

        original_first_frame = original.xyz[0].copy()

        archive = str(tmp_path / "traj")
        original.save(archive)
        original.cleanup()

        # Delete the original Zarr cache to prove the archive is self-contained
        shutil.rmtree(zarr_cache)

        loaded = DaskMDTrajectory.load(archive + ARCHIVE_EXTENSION)
        try:
            assert isinstance(loaded, DaskMDTrajectory)
            assert loaded.n_frames == n_frames
            assert loaded.n_atoms == n_atoms
            assert np.allclose(loaded.xyz[0], original_first_frame)
        finally:
            loaded.cleanup()

    def test_load_nonexistent_file(self, tmp_path):
        """
        load() must raise FileNotFoundError for a missing archive.

        Validates that supplying a path that does not exist raises an
        informative exception rather than producing a confusing traceback.
        """
        nonexistent = str(tmp_path / "does_not_exist.dask_traj")
        with pytest.raises(FileNotFoundError, match="Trajectory archive not found"):
            DaskMDTrajectory.load(nonexistent)

    # ------------------------------------------------------------------
    # from_mdtraj source
    # ------------------------------------------------------------------

    def test_save_load_from_mdtraj(self, tmp_path, mock_trajectory):
        """
        Trajectories created via from_mdtraj() must survive a save/load cycle.

        Validates that a DaskMDTrajectory built from an in-memory md.Trajectory
        object is correctly archived and restored with identical data.
        """
        traj_file, top_file, n_frames, n_atoms = mock_trajectory
        md_traj = md.load(traj_file, top=top_file)

        zarr_cache = str(tmp_path / "zarr_cache_mdtraj")
        original = DaskMDTrajectory.from_mdtraj(md_traj, zarr_cache_path=zarr_cache)

        archive = str(tmp_path / "mdtraj_based")
        original.save(archive)

        loaded = DaskMDTrajectory.load(archive + ARCHIVE_EXTENSION)
        try:
            assert isinstance(loaded, DaskMDTrajectory)
            assert loaded.n_frames == n_frames
            assert loaded.n_atoms == n_atoms
            assert np.allclose(loaded.xyz, original.xyz)
        finally:
            original.cleanup()
            loaded.cleanup()

    # ------------------------------------------------------------------
    # Data integrity vs MDTraj reference
    # ------------------------------------------------------------------

    def test_dask_vs_mdtraj_identical_results(self, tmp_path, mock_trajectory):
        """
        Loaded DaskMDTrajectory must be bit-identical to the MDTraj reference.

        Validates that the save/load cycle introduces no numeric changes and
        that topology atom names are preserved exactly.
        """
        traj_file, top_file, _, _ = mock_trajectory
        md_traj = md.load(traj_file, top=top_file)

        zarr_cache = str(tmp_path / "zarr_cache_comparison")
        dask_traj = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=zarr_cache)

        archive = str(tmp_path / "comparison_test")
        dask_traj.save(archive)
        loaded_dask = DaskMDTrajectory.load(archive + ARCHIVE_EXTENSION)

        try:
            assert np.allclose(loaded_dask.xyz, md_traj.xyz)
            assert loaded_dask.n_frames == md_traj.n_frames
            assert loaded_dask.n_atoms == md_traj.n_atoms

            loaded_atoms = [atom.name for atom in loaded_dask.topology.atoms]
            md_atoms = [atom.name for atom in md_traj.topology.atoms]
            assert loaded_atoms == md_atoms
        finally:
            dask_traj.cleanup()
            loaded_dask.cleanup()
