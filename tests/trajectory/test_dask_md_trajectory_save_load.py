# mdxplain - A Python toolkit for molecular dynamics trajectory analysis
#
# Author: Maximilian Salomon
# Created with assistance from Claude Code (Claude Sonnet 4.0)
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

"""Tests for DaskMDTrajectory save/load functionality."""

import os
import pytest
import numpy as np
import shutil
from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory
import mdtraj as md


class TestDaskMDTrajectorySaveLoad:
    """Test save/load functionality of DaskMDTrajectory."""
    
    @pytest.fixture
    def mock_trajectory(self, tmp_path):
        """Create a mock trajectory for testing."""
        # Create minimal test trajectory
        n_frames = 5
        n_atoms = 3
        
        # Create topology
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue('ALA', chain)
        for i in range(n_atoms):
            topology.add_atom(f'CA{i}', md.element.carbon, residue)
        
        # Create trajectory
        xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32)
        traj = md.Trajectory(xyz, topology)
        
        # Save to files
        traj_file = str(tmp_path / "test.xtc")
        top_file = str(tmp_path / "test.pdb")
        traj.save_xtc(traj_file)
        traj.save_pdb(top_file)
        
        return traj_file, top_file, n_frames, n_atoms
    
    def test_save_creates_directories(self, tmp_path, mock_trajectory):
        """Test that save() creates parent directories if needed."""
        traj_file, top_file, n_frames, n_atoms = mock_trajectory
        
        # Create DaskMDTrajectory
        zarr_cache = str(tmp_path / "zarr_cache")
        dask_traj = DaskMDTrajectory(
            traj_file, top_file,
            zarr_cache_path=zarr_cache
        )
        
        # Save to nested path that doesn't exist
        save_path = tmp_path / "output" / "trajectories" / "my_traj.pkl"
        assert not save_path.parent.exists()
        
        dask_traj.save(str(save_path))
        
        # Check directory was created and file exists
        assert save_path.parent.exists()
        assert save_path.exists()
    
    def test_save_load_roundtrip(self, tmp_path, mock_trajectory):
        """Test complete save/load roundtrip."""
        traj_file, top_file, n_frames, n_atoms = mock_trajectory
        
        # Create DaskMDTrajectory
        zarr_cache = str(tmp_path / "zarr_cache")
        original = DaskMDTrajectory(
            traj_file, top_file,
            zarr_cache_path=zarr_cache
        )
        
        # Save
        save_path = str(tmp_path / "saved_traj.pkl")
        original.save(save_path)
        
        # Load
        loaded = DaskMDTrajectory.load(save_path)
        
        # Verify type and properties
        assert isinstance(loaded, DaskMDTrajectory)
        assert loaded.n_frames == original.n_frames == n_frames
        assert loaded.n_atoms == original.n_atoms == n_atoms
        assert loaded.trajectory_file == original.trajectory_file
        assert loaded.topology_file == original.topology_file
        assert loaded.zarr_cache_path == original.zarr_cache_path
        
        # Verify data access works
        assert np.allclose(loaded.xyz, original.xyz)
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test that load() raises FileNotFoundError for missing files."""
        nonexistent = str(tmp_path / "does_not_exist.pkl")
        
        with pytest.raises(FileNotFoundError, match="Trajectory file not found"):
            DaskMDTrajectory.load(nonexistent)
    
    def test_zarr_cache_independence_after_save(self, tmp_path, mock_trajectory):
        """Test that saved trajectory works independently of zarr cache."""
        traj_file, top_file, n_frames, n_atoms = mock_trajectory
        
        # Create and save
        zarr_cache = str(tmp_path / "zarr_cache")
        original = DaskMDTrajectory(
            traj_file, top_file,
            zarr_cache_path=zarr_cache
        )
        
        # Get original data before saving
        original_first_frame = original.xyz[0].copy()
        
        save_path = str(tmp_path / "traj.pkl")
        original.save(save_path)
        
        # Delete zarr cache
        if os.path.exists(zarr_cache):
            shutil.rmtree(zarr_cache)
        
        # Load should work (creates object)
        loaded = DaskMDTrajectory.load(save_path)
        assert isinstance(loaded, DaskMDTrajectory)
        assert loaded.n_frames == n_frames
        
        # Data access should still work because pickle saved the complete state
        loaded_first_frame = loaded.xyz[0]
        assert np.allclose(loaded_first_frame, original_first_frame)
        
        # However, the zarr cache path still references the deleted location
        assert loaded.zarr_cache_path == zarr_cache
        assert not os.path.exists(loaded.zarr_cache_path)
    
    def test_save_load_from_mdtraj(self, tmp_path, mock_trajectory):
        """Test save/load roundtrip for DaskMDTrajectory created from MDTraj."""
        traj_file, top_file, n_frames, n_atoms = mock_trajectory
        
        # Load as MDTraj first
        md_traj = md.load(traj_file, top=top_file)
        
        # Create DaskMDTrajectory from MDTraj
        zarr_cache = str(tmp_path / "zarr_cache_mdtraj")
        original = DaskMDTrajectory.from_mdtraj(
            md_traj, zarr_cache_path=zarr_cache
        )
        
        # Save
        save_path = str(tmp_path / "mdtraj_based.pkl")
        original.save(save_path)
        
        # Load
        loaded = DaskMDTrajectory.load(save_path)
        
        # Verify type and properties
        assert isinstance(loaded, DaskMDTrajectory)
        assert loaded.n_frames == original.n_frames == n_frames
        assert loaded.n_atoms == original.n_atoms == n_atoms
        assert loaded.zarr_cache_path == original.zarr_cache_path
        
        # Verify data access works
        assert np.allclose(loaded.xyz, original.xyz)

    def test_dask_vs_mdtraj_identical_results(self, tmp_path, mock_trajectory):
        """Test that saved/loaded DaskMDTrajectory gives identical results to MDTraj."""
        traj_file, top_file, n_frames, n_atoms = mock_trajectory
        
        # Load reference MDTraj
        md_traj = md.load(traj_file, top=top_file)
        
        # Create, save and load DaskMDTrajectory from files
        zarr_cache = str(tmp_path / "zarr_cache_comparison")
        dask_traj = DaskMDTrajectory(
            traj_file, top_file,
            zarr_cache_path=zarr_cache
        )
        
        save_path = str(tmp_path / "comparison_test.pkl")
        dask_traj.save(save_path)
        loaded_dask = DaskMDTrajectory.load(save_path)
        
        # Compare xyz coordinates
        assert np.allclose(loaded_dask.xyz, md_traj.xyz)
        
        # Compare basic properties
        assert loaded_dask.n_frames == md_traj.n_frames
        assert loaded_dask.n_atoms == md_traj.n_atoms
        
        # Compare topology (atom names)
        loaded_atoms = [atom.name for atom in loaded_dask.topology.atoms]
        md_atoms = [atom.name for atom in md_traj.topology.atoms]
        assert loaded_atoms == md_atoms
