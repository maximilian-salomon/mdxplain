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
Tests for DaskMDTrajectory data management.

Validates the new behavior for strict in-place overwrites, permanent
file creation for non-inplace operations, and fresh initialization logic.
"""

import os
import pytest
import numpy as np
import mdtraj as md
import zarr
from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory

class TestDaskMDTrajectoryDataManagement:
    """Test suite for DaskMDTrajectory data management policies."""

    @pytest.fixture
    def sample_trajectory(self, tmp_path):
        """
        Create a simple MDTraj trajectory and save it to disk.
        
        Returns
        -------
        tuple
            (path_to_xtc, path_to_pdb)
        """
        np.random.seed(42)
        n_frames = 10
        n_atoms = 20
        xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain)
        for _ in range(n_atoms):
            topology.add_atom("C", md.element.carbon, residue)
        
        traj = md.Trajectory(xyz, topology)
        
        traj_file = tmp_path / "test_traj.xtc"
        top_file = tmp_path / "test_traj.pdb"
        
        traj.save(str(traj_file))
        traj[0].save(str(top_file))
        
        return str(traj_file), str(top_file)

    def test_initialization_overwrite(self, sample_trajectory, tmp_path):
        """
        Test that initializing from source overwrites existing Zarr cache.
        
        Verifies that if a Zarr cache already exists at the target location,
        it is deleted and recreated fresh, preventing usage of stale data.
        """
        traj_file, top_file = sample_trajectory
        cache_path = tmp_path / "cache.zarr"
        
        # 1. First initialization
        dt1 = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path))
        assert os.path.exists(cache_path)
        
        # Capture data
        data1 = dt1.xyz.copy()
        
        # Modify the cache to leave a mark (add a dummy file)
        marker_file = cache_path / "marker.txt"
        with open(marker_file, "w") as f:
            f.write("I was here")
        assert marker_file.exists()
        
        # cleanup reference to close files
        del dt1
        
        # 2. Second initialization
        dt2 = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path))
        
        # Verify marker is GONE (directory was recreated)
        assert not marker_file.exists()
        assert os.path.exists(cache_path)
        
        # Verify data is identical (recreated from same source)
        assert np.allclose(dt2.xyz, data1)

    def test_inplace_false_creates_new_file(self, sample_trajectory, tmp_path):
        """
        Test that inplace=False creates a new file and preserves original.
        
        Verifies that non-inplace operations (like atom_slice) generate a
        new Zarr file with a distinct name and do not modify the original
        source trajectory's cache.
        """
        traj_file, top_file = sample_trajectory
        cache_path = tmp_path / "original.zarr"
        
        dt = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path))
        original_coords = dt.xyz.copy()
        
        # Operation: Slice first 5 atoms
        indices = np.arange(5)
        sliced = dt.atom_slice(indices, inplace=False)
        
        # Verify paths
        assert dt.zarr_cache_path == str(cache_path)
        assert sliced.zarr_cache_path != str(cache_path)
        assert "atom_slice" in sliced.zarr_cache_path
        
        # Verify both files exist
        assert os.path.exists(sliced.zarr_cache_path)
        assert os.path.exists(cache_path)
        
        # Verify content
        # Original should be untouched
        assert np.allclose(dt.xyz, original_coords)
        
        # Sliced should contain subset
        expected_sliced = original_coords[:, indices, :]
        assert np.allclose(sliced.xyz, expected_sliced)

    def test_inplace_true_overwrites_atomic(self, sample_trajectory, tmp_path):
        """
        Test that inplace=True overwrites the original file atomically.
        
        Verifies that inplace operations replace the original Zarr cache
        with the processed result, update the object's attributes, and
        leave no temporary swap files behind.
        """
        traj_file, top_file = sample_trajectory
        cache_path = tmp_path / "to_overwrite.zarr"
        
        dt = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path))
        original_path = dt.zarr_cache_path
        original_coords = dt.xyz.copy()
        
        # Operation: Slice first 5 atoms inplace
        indices = np.arange(5)
        dt.atom_slice(indices, inplace=True)
        
        # Checks
        assert dt.zarr_cache_path == original_path
        assert os.path.exists(original_path)
        
        # Verify no swap file left
        assert not os.path.exists(original_path + ".swap")
        
        # Verify data persistence by opening store directly
        # Should now contain only sliced atoms
        expected_coords = original_coords[:, indices, :]
        
        store = zarr.open(original_path, mode='r')
        stored_coords = store['coordinates'][:]
        assert np.allclose(stored_coords, expected_coords)
        
        # Verify object state
        assert np.allclose(dt.xyz, expected_coords)

    def test_chained_operations_no_dataloss(self, sample_trajectory, tmp_path):
        """
        Test that chaining non-inplace operations does not cause data loss.
        
        Fixes a specific bug where chaining operations (slice -> slice)
        on temporary stores would prematurely delete the intermediate
        store. Verifies that all intermediate results persist.
        """
        traj_file, top_file = sample_trajectory
        cache_path = tmp_path / "chain.zarr"
        
        dt = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path))
        original_coords = dt.xyz.copy()
        
        # Chain: Slice 1 (10 atoms) -> Slice 2 (5 atoms)
        indices1 = np.arange(10)
        slice1 = dt.atom_slice(indices1, inplace=False)
        
        indices2 = np.arange(5)
        slice2 = slice1.atom_slice(indices2, inplace=False)
        
        # Verify slice1 still has valid data
        expected_slice1 = original_coords[:, indices1, :]
        assert np.allclose(slice1.xyz, expected_slice1)
        
        # Verify slice2 is correct (subset of slice1)
        expected_slice2 = expected_slice1[:, indices2, :]
        assert np.allclose(slice2.xyz, expected_slice2)
        
        # Verify both files exist on disk
        assert os.path.exists(slice1.zarr_cache_path)
        assert os.path.exists(slice2.zarr_cache_path)

    def test_join_creates_new_file(self, sample_trajectory, tmp_path):
        """
        Test that join creates a new permanent file.
        
        Verifies that the join operation generates a new Zarr file
        and does not modify the inputs.
        """
        traj_file, top_file = sample_trajectory
        cache_path1 = tmp_path / "traj1.zarr"
        cache_path2 = tmp_path / "traj2.zarr"
        
        t1 = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path1))
        t2 = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path2))
        
        t1_coords = t1.xyz.copy()
        t2_coords = t2.xyz.copy()
        
        joined = t1.join(t2)
        
        # Verify paths
        assert joined.zarr_cache_path != t1.zarr_cache_path
        assert joined.zarr_cache_path != t2.zarr_cache_path
        assert "join" in joined.zarr_cache_path
        assert os.path.exists(joined.zarr_cache_path)
        
        # Verify content
        expected_coords = np.concatenate([t1_coords, t2_coords], axis=0)
        assert np.allclose(joined.xyz, expected_coords)

    def test_stack_creates_new_file(self, sample_trajectory, tmp_path):
        """
        Test that stack creates a new permanent file.
        
        Verifies that the stack operation generates a new Zarr file
        and does not modify the inputs.
        """
        traj_file, top_file = sample_trajectory
        cache_path1 = tmp_path / "traj1.zarr"
        cache_path2 = tmp_path / "traj2.zarr"
        
        t1 = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path1))
        t2 = DaskMDTrajectory(traj_file, top_file, zarr_cache_path=str(cache_path2))
        
        t1_coords = t1.xyz.copy()
        t2_coords = t2.xyz.copy()
        
        stacked = t1.stack(t2)
        
        # Verify paths
        assert stacked.zarr_cache_path != t1.zarr_cache_path
        assert stacked.zarr_cache_path != t2.zarr_cache_path
        assert "stack" in stacked.zarr_cache_path
        assert os.path.exists(stacked.zarr_cache_path)
        
        # Verify content
        expected_coords = np.concatenate([t1_coords, t2_coords], axis=1)
        assert np.allclose(stacked.xyz, expected_coords)
