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
Test file for TrajectoryManager.superpose() functionality.

Tests verify that superpose works correctly:
- In-place modification of trajectories
- RMSD reduction after alignment
- Correct handling of different atom selections
- Parameter validation
- DaskMDTrajectory support with chunk-wise processing
"""

import pytest
import numpy as np
import mdtraj as md
import tempfile
import os
from unittest.mock import patch
import shutil

from mdxplain.pipeline.entities.pipeline_data import PipelineData
from mdxplain.trajectory.manager.trajectory_manager import TrajectoryManager
from mdxplain.trajectory.entities.trajectory_data import TrajectoryData
from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory


def create_test_trajectory(n_frames=10, n_atoms=80, add_drift=False):
    """
    Create a realistic test trajectory with protein-like structure.

    Parameters:
    -----------
    n_frames : int
        Number of frames
    n_atoms : int
        Number of atoms (must be multiple of 4 for proper residues)
    add_drift : bool
        If True, add artificial drift to simulate unaligned trajectory

    Returns:
    --------
    md.Trajectory
        Test trajectory with proper backbone topology and realistic coordinates
    """
    # Create topology with backbone atoms
    topology = md.Topology()
    chain = topology.add_chain()

    # Create residues with N, CA, C, O atoms each
    n_residues = n_atoms // 4
    for i in range(n_residues):
        residue = topology.add_residue(f"ALA", chain, i)
        topology.add_atom("N", element=md.element.nitrogen, residue=residue)
        topology.add_atom("CA", element=md.element.carbon, residue=residue)
        topology.add_atom("C", element=md.element.carbon, residue=residue)
        topology.add_atom("O", element=md.element.oxygen, residue=residue)

    # Create realistic protein-like coordinates
    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

    # Build initial alpha-helix-like structure
    for res_idx in range(n_residues):
        base_idx = res_idx * 4

        # Alpha helix parameters: 3.6 residues per turn, 1.5 Å rise per residue
        angle = res_idx * 2 * np.pi / 3.6  # 100 degrees per residue
        z_pos = res_idx * 1.5  # 1.5 Å rise per residue
        radius = 2.3  # Approximate radius for alpha helix

        # Initial coordinates for first frame
        xyz[0, base_idx] = [radius * np.cos(angle), radius * np.sin(angle), z_pos]  # N
        xyz[0, base_idx + 1] = [radius * np.cos(angle + 0.1), radius * np.sin(angle + 0.1), z_pos + 0.5]  # CA
        xyz[0, base_idx + 2] = [radius * np.cos(angle + 0.2), radius * np.sin(angle + 0.2), z_pos + 1.0]  # C
        xyz[0, base_idx + 3] = [radius * np.cos(angle + 0.3), radius * np.sin(angle + 0.3), z_pos + 1.2]  # O

    # Copy initial structure to all frames
    for frame in range(1, n_frames):
        xyz[frame] = xyz[0].copy()

    if add_drift:
        # Add significant realistic drift - both translation and rotation
        for frame in range(n_frames):
            if frame == 0:
                continue

            # Progressive translation drift (larger for testability)
            translation = np.array([frame * 2.0, frame * 1.5, frame * 1.0])

            # Progressive rotation around multiple axes (larger for testability)
            angle_x = frame * 0.15  # Significant rotation around x
            angle_y = frame * 0.20  # Significant rotation around y
            angle_z = frame * 0.10  # Significant rotation around z

            # Rotation matrices
            cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
            cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)

            rot_x = np.array([[1, 0, 0],
                             [0, cos_x, -sin_x],
                             [0, sin_x, cos_x]])

            rot_y = np.array([[cos_y, 0, sin_y],
                             [0, 1, 0],
                             [-sin_y, 0, cos_y]])

            rot_z = np.array([[cos_z, -sin_z, 0],
                             [sin_z, cos_z, 0],
                             [0, 0, 1]])

            # Combined rotation
            rotation = rot_z @ rot_y @ rot_x

            # Apply rotation and translation
            for atom in range(n_atoms):
                xyz[frame, atom] = rotation @ xyz[frame, atom] + translation

    trajectory = md.Trajectory(xyz, topology)
    return trajectory


def create_dask_test_trajectory(
    n_frames=10,
    n_atoms=80,
    add_drift=False,
    chunk_size=3,
    base_dir=None,
):
    """
    Create a DaskMDTrajectory for testing.

    Parameters:
    -----------
    n_frames : int
        Number of frames
    n_atoms : int
        Number of atoms (must be multiple of 4 for proper residues)
    add_drift : bool
        If True, add artificial drift to simulate unaligned trajectory
    chunk_size : int
        Chunk size for DaskMDTrajectory

    Returns:
    --------
    DaskMDTrajectory
        Test trajectory with Zarr backend
    """
    # Create regular trajectory first
    regular_traj = create_test_trajectory(n_frames, n_atoms, add_drift)

    # Create temporary directory and file
    temp_dir = tempfile.mkdtemp(dir=str(base_dir) if base_dir is not None else None)
    temp_trajectory_file = os.path.join(temp_dir, "test_trajectory.xtc")
    temp_topology_file = os.path.join(temp_dir, "test_topology.pdb")

    # Save trajectory and topology
    regular_traj.save_xtc(temp_trajectory_file)
    regular_traj.save_pdb(temp_topology_file)

    # Create DaskMDTrajectory
    dask_traj = DaskMDTrajectory(
        trajectory_file=temp_trajectory_file,
        topology_file=temp_topology_file,
        chunk_size=chunk_size
    )

    # Store cleanup info
    dask_traj._test_temp_dir = temp_dir

    return dask_traj


def cleanup_dask_trajectory(dask_traj):
    """Clean up temporary files created for DaskMDTrajectory."""
    temp_dir = getattr(dask_traj, '_test_temp_dir', None)
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def dask_traj_factory(tmp_path_factory):
    """Factory fixture that tracks created DaskMDTrajectory instances for cleanup."""
    created = []

    def factory(**kwargs):
        cache_dir = tmp_path_factory.mktemp("dask_traj")
        traj = create_dask_test_trajectory(base_dir=str(cache_dir), **kwargs)
        created.append(traj)
        return traj

    def register(traj):
        created.append(traj)

    yield factory, register

    for traj in created:
        cleanup_dask_trajectory(traj)


class TestSuperpose:
    """Test class for superpose functionality."""

    def test_superpose_is_inplace(self):
        """Verify superpose modifies trajectories in-place."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        # Create trajectory with drift
        traj = create_test_trajectory(n_frames=5, n_atoms=80, add_drift=True)
        original_xyz_id = id(traj.xyz)
        original_coords = traj.xyz.copy()
        # Measure raw RMSD to reference frame without additional alignment
        pre_rmsd = np.sqrt(np.mean((original_coords - original_coords[0]) ** 2, axis=(1, 2)))

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj]
        pipeline_data.trajectory_data.trajectory_names = ["test_traj"]

        # Superpose
        traj_manager.superpose(
            pipeline_data,
            traj_selection=0,
            reference_traj=0,
            reference_frame=0,
            atom_selection="all"
        )

        # Verify in-place modification
        assert id(traj.xyz) == original_xyz_id, "Trajectory should be modified in-place"
        assert not np.allclose(original_coords, traj.xyz), "Coordinates should have changed"

        post_rmsd = md.rmsd(traj, traj[0], superpose=False)
        assert np.max(post_rmsd[1:]) < 1e-3, "Frames should align to reference within tolerance"
        assert np.max(post_rmsd[1:]) < np.max(pre_rmsd[1:]) * 1e-3, "RMSD should drop several orders of magnitude"

    def test_mdtraj_dask_equivalence_direct(self, tmp_path):
        """DaskMDTrajectory and MDTraj must produce identical results (no pipeline)."""
        # Create test trajectory with drift
        traj = create_test_trajectory(n_frames=8, n_atoms=80, add_drift=True)
        xtc_path = tmp_path / 'test_equiv.xtc'
        pdb_path = tmp_path / 'test_equiv.pdb'

        traj.save_xtc(str(xtc_path))
        traj[0].save(str(pdb_path))

        # Test 1: Basic superpose to frame 0
        md_traj1 = md.load(str(xtc_path), top=str(pdb_path))
        dask_traj1 = DaskMDTrajectory(
            str(xtc_path),
            str(pdb_path),
            zarr_cache_path=str(tmp_path / 'test_equiv_chunk1.zarr')
        )

        md_traj1.superpose(md_traj1[0])
        dask_traj1.superpose(dask_traj1[0])

        assert np.allclose(md_traj1.xyz, dask_traj1.xyz, atol=1e-6), \
            "MDTraj and DaskMDTrajectory should produce identical results"

        # Test 2: Different reference frame
        md_traj2 = md.load(str(xtc_path), top=str(pdb_path))
        dask_traj2 = DaskMDTrajectory(
            str(xtc_path),
            str(pdb_path),
            zarr_cache_path=str(tmp_path / 'test_equiv_chunk2.zarr')
        )

        md_traj2.superpose(md_traj2[5])
        dask_traj2.superpose(dask_traj2[5])

        assert np.allclose(md_traj2.xyz, dask_traj2.xyz, atol=1e-6), \
            "Results should match for different reference frames"

        # Test 3: With atom indices
        md_traj3 = md.load(str(xtc_path), top=str(pdb_path))
        dask_traj3 = DaskMDTrajectory(
            str(xtc_path),
            str(pdb_path),
            zarr_cache_path=str(tmp_path / 'test_equiv_chunk3.zarr')
        )

        ca_indices = md_traj3.topology.select("name CA")

        md_traj3.superpose(md_traj3[0], atom_indices=ca_indices)
        dask_traj3.superpose(dask_traj3[0], atom_indices=ca_indices)

        assert np.allclose(md_traj3.xyz, dask_traj3.xyz, atol=1e-6), \
            "Results should match with atom selection"

        # Cleanup

    def test_superpose_reference_frame_unchanged(self):
        """Reference frame should remain unchanged after superposition."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        traj = create_test_trajectory(n_frames=6, n_atoms=80, add_drift=True)
        reference_frame_coords = traj.xyz[0].copy()  # Save frame 0 coordinates

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj]
        pipeline_data.trajectory_data.trajectory_names = ["ref_test"]

        # Superpose to frame 0
        traj_manager.superpose(
            pipeline_data,
            traj_selection=0,
            reference_traj=0,
            reference_frame=0,
            atom_selection="all"
        )

        # Verify reference frame unchanged
        assert np.allclose(traj.xyz[0], reference_frame_coords, atol=1e-6), \
            "Reference frame coordinates should remain unchanged"

    def test_superpose_with_atom_selection(self):
        """Test different atom selections work correctly and produce different results."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        base_traj = create_test_trajectory(n_frames=6, n_atoms=80, add_drift=True)

        # Test different atom selections
        test_selections = [
            "all",
            "name CA",
            "backbone",  # backbone atoms
        ]

        results = {}

        for selection in test_selections:
            # Create fresh copy for each test
            test_traj = base_traj.slice(range(base_traj.n_frames))
            original_traj = base_traj.slice(range(base_traj.n_frames))

            # Setup pipeline data
            pipeline_data.trajectory_data = TrajectoryData()
            pipeline_data.trajectory_data.trajectories = [test_traj]
            pipeline_data.trajectory_data.trajectory_names = [f"selection_test_{selection}"]

            # Get atom indices for this selection
            align_atom_indices = test_traj.topology.select(selection)

            # Perform superposition
            traj_manager.superpose(
                pipeline_data,
                traj_selection=0,
                reference_traj=0,
                reference_frame=0,
                atom_selection=selection
            )

            # Store final coordinates for comparison
            results[selection] = {
                'n_align_atoms': len(align_atom_indices),
                'final_coords': test_traj.xyz.copy()
            }

            # Test 2: Trajectory should be modified
            assert not np.allclose(original_traj.xyz, test_traj.xyz, atol=1e-6), \
                f"Trajectory should be modified for selection '{selection}'"

        # Test: atom_selection works - different selections give different results
        assert not np.allclose(results["all"]["final_coords"], results["name CA"]["final_coords"], atol=1e-6), \
            "ALL and CA selections should produce different final coordinates"

    def test_superpose_validation(self):
        """Test parameter validation."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        # Test with empty trajectory data
        with pytest.raises(ValueError, match="No trajectories loaded"):
            traj_manager.superpose(pipeline_data, traj_selection=0)

        # Setup with one trajectory
        traj = create_test_trajectory(n_frames=5, n_atoms=80)
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj]
        pipeline_data.trajectory_data.trajectory_names = ["validation_test"]

        # Test invalid reference trajectory
        with pytest.raises(ValueError, match="Reference trajectory index 5 is invalid"):
            traj_manager.superpose(pipeline_data, traj_selection=0, reference_traj=5)

        # Test invalid reference frame
        with pytest.raises(ValueError, match="Reference frame index 100 is invalid"):
            traj_manager.superpose(pipeline_data, traj_selection=0, reference_frame=100)

        # Test invalid atom selection
        with pytest.raises(ValueError, match="Invalid atom selection"):
            traj_manager.superpose(pipeline_data, traj_selection=0, atom_selection="invalid_selection_string")

        # Test empty atom selection
        with pytest.raises(ValueError, match="produced no atoms"):
            traj_manager.superpose(pipeline_data, traj_selection=0, atom_selection="name XYZ")  # Non-existent atom

    def test_superpose_selection_subset(self):
        """Only selected trajectories should be aligned while others stay untouched."""
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        traj1 = create_test_trajectory(n_frames=5, n_atoms=80, add_drift=True)
        traj2 = create_test_trajectory(n_frames=5, n_atoms=80, add_drift=True)

        original_traj1 = traj1.xyz.copy()
        original_traj2 = traj2.xyz.copy()
        pre_rmsd = np.sqrt(np.mean((original_traj1 - original_traj1[0]) ** 2, axis=(1, 2)))

        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj1, traj2]
        pipeline_data.trajectory_data.trajectory_names = ["sel_a", "sel_b"]

        traj_manager.superpose(
            pipeline_data,
            reference_traj=0,
            reference_frame=0,
            traj_selection=["sel_a"],
            atom_selection="all"
        )

        post_rmsd = md.rmsd(traj1, traj1[0], superpose=False)
        assert np.max(post_rmsd[1:]) < 1e-3, "Selected trajectory should align to reference"
        assert np.max(post_rmsd[1:]) < np.max(pre_rmsd[1:]) * 1e-3, "Alignment should significantly improve"

        assert np.allclose(traj2.xyz, original_traj2), "Unselected trajectory must remain unchanged"

    def test_superpose_raises_on_atom_mismatch(self):
        """Atom count mismatches between trajectories should raise a clear error."""
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        ref_traj = create_test_trajectory(n_frames=4, n_atoms=80, add_drift=True)
        shorter_traj = create_test_trajectory(n_frames=4, n_atoms=76, add_drift=True)

        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [ref_traj, shorter_traj]
        pipeline_data.trajectory_data.trajectory_names = ["ref", "shorter"]

        with pytest.raises(ValueError, match="Atom count mismatch"):
            traj_manager.superpose(
                pipeline_data,
                reference_traj=0,
                reference_frame=0,
                traj_selection="all",
                atom_selection="all"
            )

    def test_superpose_multiple_trajectories(self):
        """Test basic functionality of superposing multiple MDTraj trajectories."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager()

        # Create multiple trajectories with same topology but different coordinates
        np.random.seed(42)
        traj1 = create_test_trajectory(n_frames=4, n_atoms=48, add_drift=True)
        traj2 = create_test_trajectory(n_frames=4, n_atoms=48, add_drift=True)

        # Store original copies for comparison
        orig_coords1 = traj1.xyz.copy()
        orig_coords2 = traj2.xyz.copy()

        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [traj1, traj2]
        pipeline_data.trajectory_data.trajectory_names = ["multi_traj1", "multi_traj2"]

        # Superpose all trajectories to first frame of first trajectory using CA atoms
        traj_manager.superpose(
            pipeline_data,
            reference_traj=0,
            reference_frame=0,
            traj_selection="all",
            atom_selection="name CA"
        )

        # Test 1: CA atoms should be perfectly aligned in reference frame
        ca_indices = traj1.topology.select("name CA")
        rmsd_traj1_ca = md.rmsd(traj1, traj1[0], atom_indices=ca_indices, superpose=False)
        assert rmsd_traj1_ca[0] < 1e-6, "Reference trajectory CA atoms should have zero RMSD"

        # Test 2: Both trajectories were actually modified
        assert not np.allclose(orig_coords1, traj1.xyz, atol=1e-6), "Trajectory 1 coordinates should have changed"
        assert not np.allclose(orig_coords2, traj2.xyz, atol=1e-6), "Trajectory 2 coordinates should have changed"

    # ===============================
    # DaskMDTrajectory Tests (Duplicated from above)
    # ===============================

    def test_dask_superpose_is_inplace(self, dask_traj_factory):
        """Verify DaskMDTrajectory superpose modifies trajectories in-place."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager(use_memmap=True)

        # Create DaskMDTrajectory with drift
        factory, _ = dask_traj_factory
        dask_traj = factory(n_frames=5, n_atoms=80, add_drift=True, chunk_size=2)
        original_id = id(dask_traj)
        original_coords = dask_traj.xyz.copy()

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [dask_traj]
        pipeline_data.trajectory_data.trajectory_names = ["dask_test_traj"]

        # Superpose
        traj_manager.superpose(
            pipeline_data,
            traj_selection=0,
            reference_traj=0,
            reference_frame=0,
            atom_selection="all"
        )

        # Verify in-place modification
        assert id(dask_traj) == original_id, "Object identity should be preserved"
        assert not np.allclose(original_coords, dask_traj.xyz), "Coordinates should have changed"


    def test_trajectory_manager_both_types(self, tmp_path):
        """TrajectoryManager should work equally well with MDTraj and DaskMDTrajectory."""
        # Create test trajectory
        traj = create_test_trajectory(n_frames=6, n_atoms=40, add_drift=True)
        xtc_path = tmp_path / 'test_both_types.xtc'
        pdb_path = tmp_path / 'test_both_types.pdb'
        zarr_path = tmp_path / 'test_both_types.zarr'

        traj.save_xtc(str(xtc_path))
        traj[0].save(str(pdb_path))

        # Load as both types
        md_traj = md.load(str(xtc_path), top=str(pdb_path))
        dask_traj = DaskMDTrajectory(
            str(xtc_path),
            str(pdb_path),
            zarr_cache_path=str(zarr_path)
        )

        # Store originals for comparison
        md_orig = md_traj.xyz.copy()
        dask_orig = dask_traj.xyz.copy()

        # Test 1: MDTraj through TrajectoryManager
        pipeline_md = PipelineData()
        pipeline_md.trajectory_data = TrajectoryData()
        pipeline_md.trajectory_data.trajectories = [md_traj]
        pipeline_md.trajectory_data.trajectory_names = ["md_test"]

        traj_manager1 = TrajectoryManager()
        traj_manager1.superpose(pipeline_md, traj_selection=0, reference_traj=0, reference_frame=0)

        # Test 2: DaskMDTrajectory through TrajectoryManager
        pipeline_dask = PipelineData()
        pipeline_dask.trajectory_data = TrajectoryData()
        pipeline_dask.trajectory_data.trajectories = [dask_traj]
        pipeline_dask.trajectory_data.trajectory_names = ["dask_test"]

        traj_manager2 = TrajectoryManager(use_memmap=True)
        traj_manager2.superpose(pipeline_dask, traj_selection=0, reference_traj=0, reference_frame=0)

        # Both should have changed
        assert not np.allclose(md_orig, md_traj.xyz), "MDTraj should have changed"
        assert not np.allclose(dask_orig, dask_traj.xyz), "DaskMDTrajectory should have changed"

        # Results should be identical
        assert np.allclose(md_traj.xyz, dask_traj.xyz, atol=1e-6), \
            "MDTraj and DaskMDTrajectory should produce identical results through TrajectoryManager"

        # Reference frames should be aligned
        assert md.rmsd(md_traj, md_traj[0], superpose=False)[0] < 1e-6, "MDTraj reference frame should be aligned"
        assert md.rmsd(dask_traj, dask_traj[0], superpose=False)[0] < 1e-6, "DaskMDTrajectory reference frame should be aligned"


    def test_dask_superpose_reference_frame_unchanged(self, dask_traj_factory):
        """Reference frame should remain unchanged after DaskMDTrajectory superposition."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager(use_memmap=True)

        factory, _ = dask_traj_factory
        dask_traj = factory(n_frames=6, n_atoms=80, add_drift=True, chunk_size=2)

        reference_frame_coords = dask_traj.xyz[0].copy()  # Save frame 0 coordinates

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [dask_traj]
        pipeline_data.trajectory_data.trajectory_names = ["dask_ref_test"]

        # Superpose to frame 0
        traj_manager.superpose(
            pipeline_data,
            traj_selection=0,
            reference_traj=0,
            reference_frame=0,
            atom_selection="all"
        )

        # Verify reference frame unchanged
        assert np.allclose(dask_traj.xyz[0], reference_frame_coords, atol=1e-6), \
            "Reference frame coordinates should remain unchanged"

    def test_dask_superpose_with_atom_selection(self, dask_traj_factory):
        """Test different atom selections work correctly with DaskMDTrajectory and produce different results."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager(use_memmap=True)

        # Test different atom selections
        test_selections = [
            "all",
            "name CA",
            "backbone",  # backbone atoms
        ]

        results = {}
        factory, _ = dask_traj_factory

        for selection in test_selections:
            # Create fresh DaskMDTrajectory for each test
            dask_traj = factory(n_frames=6, n_atoms=80, add_drift=True, chunk_size=2)

            # Store original coordinates for comparison
            original_coords = dask_traj.xyz.copy()

            # Setup pipeline data
            pipeline_data.trajectory_data = TrajectoryData()
            pipeline_data.trajectory_data.trajectories = [dask_traj]
            pipeline_data.trajectory_data.trajectory_names = [f"dask_selection_test_{selection}"]

            # Get atom indices for this selection
            align_atom_indices = dask_traj.topology.select(selection)

            # Perform superposition
            traj_manager.superpose(
                pipeline_data,
                traj_selection=0,
                reference_traj=0,
                reference_frame=0,
                atom_selection=selection
            )

            # Store final coordinates for comparison
            results[selection] = {
                'n_align_atoms': len(align_atom_indices),
                'final_coords': dask_traj.xyz.copy()
            }

            # Test 2: Coordinates should have changed
            assert not np.allclose(original_coords, dask_traj.xyz, atol=1e-6), \
                f"DaskMDTrajectory coordinates should change for selection '{selection}'"

        # Test: atom_selection works - different selections give different results
        assert not np.allclose(results["all"]["final_coords"], results["name CA"]["final_coords"], atol=1e-6), \
            "DaskMD: ALL and CA selections should produce different final coordinates"

    def test_dask_superpose_validation(self, dask_traj_factory):
        """Test parameter validation with DaskMDTrajectory."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager(use_memmap=True)

        # Test with empty trajectory data
        with pytest.raises(ValueError, match="No trajectories loaded"):
            traj_manager.superpose(pipeline_data, traj_selection=0)

        # Setup with one DaskMDTrajectory
        factory, _ = dask_traj_factory
        dask_traj = factory(n_frames=5, n_atoms=80, chunk_size=2)

        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [dask_traj]
        pipeline_data.trajectory_data.trajectory_names = ["dask_validation_test"]

        # Test invalid reference trajectory
        with pytest.raises(ValueError, match="Reference trajectory index 5 is invalid"):
            traj_manager.superpose(pipeline_data, traj_selection=0, reference_traj=5)

        # Test invalid reference frame
        with pytest.raises(ValueError, match="Reference frame index 100 is invalid"):
            traj_manager.superpose(pipeline_data, traj_selection=0, reference_frame=100)

        # Test invalid atom selection
        with pytest.raises(ValueError, match="Invalid atom selection"):
            traj_manager.superpose(pipeline_data, traj_selection=0, atom_selection="invalid_selection_string")

        # Test empty atom selection
        with pytest.raises(ValueError, match="produced no atoms"):
            traj_manager.superpose(pipeline_data, traj_selection=0, atom_selection="name XYZ")  # Non-existent atom

    def test_multiple_trajectory_equivalence(self, dask_traj_factory):
        """Test that TrajectoryManager works correctly with multiple trajectories for both types."""
        # Setup
        pipeline_data_md = PipelineData()
        pipeline_data_dask = PipelineData()
        traj_manager = TrajectoryManager(use_memmap=True)

        # Create test trajectories
        np.random.seed(42)
        md_traj1 = create_test_trajectory(n_frames=4, n_atoms=48, add_drift=True)
        md_traj2 = create_test_trajectory(n_frames=4, n_atoms=48, add_drift=True)

        np.random.seed(42)
        factory, _ = dask_traj_factory
        dask_traj1 = factory(n_frames=4, n_atoms=48, add_drift=True, chunk_size=2)
        dask_traj2 = factory(n_frames=4, n_atoms=48, add_drift=True, chunk_size=2)

        # Test MDTraj functionality
        pipeline_data_md.trajectory_data = TrajectoryData()
        pipeline_data_md.trajectory_data.trajectories = [md_traj1, md_traj2]
        pipeline_data_md.trajectory_data.trajectory_names = ["md_traj1", "md_traj2"]

        orig_md_coords1 = md_traj1.xyz.copy()
        orig_md_coords2 = md_traj2.xyz.copy()

        traj_manager.superpose(
            pipeline_data_md,
            reference_traj=0,
            reference_frame=0,
            traj_selection="all",
            atom_selection="name CA"
        )

        # Test DaskMDTrajectory functionality
        pipeline_data_dask.trajectory_data = TrajectoryData()
        pipeline_data_dask.trajectory_data.trajectories = [dask_traj1, dask_traj2]
        pipeline_data_dask.trajectory_data.trajectory_names = ["dask_traj1", "dask_traj2"]

        orig_dask_coords1 = dask_traj1.xyz.copy()
        orig_dask_coords2 = dask_traj2.xyz.copy()

        traj_manager.superpose(
            pipeline_data_dask,
            reference_traj=0,
            reference_frame=0,
            traj_selection="all",
            atom_selection="name CA"
        )

        # Both should work without error and modify coordinates
        assert not np.allclose(orig_md_coords1, md_traj1.xyz, atol=1e-6), "MDTraj trajectory 1 should be modified"
        assert not np.allclose(orig_md_coords2, md_traj2.xyz, atol=1e-6), "MDTraj trajectory 2 should be modified"
        assert not np.allclose(orig_dask_coords1, dask_traj1.xyz, atol=1e-6), "DaskMDTrajectory 1 should be modified"
        assert not np.allclose(orig_dask_coords2, dask_traj2.xyz, atol=1e-6), "DaskMDTrajectory 2 should be modified"

    def test_chunking_consistency(self, dask_traj_factory):
        """Test that different chunk sizes produce identical results."""
        # Create trajectories with different chunk sizes
        np.random.seed(42)
        factory, _ = dask_traj_factory
        dask_traj_chunk2 = factory(n_frames=8, n_atoms=48, add_drift=True, chunk_size=2)
        np.random.seed(42)  # Same seed for identical starting data
        dask_traj_chunk4 = factory(n_frames=8, n_atoms=48, add_drift=True, chunk_size=4)

        # Superpose both with identical parameters
        dask_traj_chunk2.superpose()  # default: reference=None, frame=0
        dask_traj_chunk4.superpose()

        # Results should be identical regardless of chunk size
        assert np.allclose(dask_traj_chunk2.xyz, dask_traj_chunk4.xyz, atol=1e-6), \
            "Different chunk sizes should produce identical superpose results"

    # ===============================
    # DaskMDTrajectory-Specific Tests (+3)
    # ===============================

    def test_dask_trajectory_zarr_files_updated(self, dask_traj_factory):
        """Verify that Zarr files are updated after superpose."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager(use_memmap=True)

        factory, register = dask_traj_factory
        dask_traj = factory(n_frames=5, n_atoms=80, add_drift=True, chunk_size=2)

        # Get zarr store path
        zarr_path = dask_traj.zarr_cache_path

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [dask_traj]
        pipeline_data.trajectory_data.trajectory_names = ["zarr_test"]

        # Superpose
        traj_manager.superpose(
            pipeline_data,
            traj_selection=0,
            reference_traj=0,
            reference_frame=0,
            atom_selection="all"
        )

        # Verify data persistence by reloading from same zarr store
        new_dask_traj = DaskMDTrajectory(
            trajectory_file=dask_traj.trajectory_file,
            topology_file=dask_traj.topology_file,
            zarr_cache_path=zarr_path,
            chunk_size=dask_traj.chunk_size
        )
        register(new_dask_traj)

        # The reloaded trajectory should have the aligned coordinates
        rmsd_reloaded = md.rmsd(new_dask_traj, new_dask_traj[0], superpose=False)
        assert rmsd_reloaded[0] < 1e-4, "Reloaded trajectory should preserve alignment"

    def test_dask_trajectory_chunk_calls_with_mock(self, dask_traj_factory):
        """Verify superpose is called once per chunk using mock."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager(use_memmap=True)

        # Create DaskMDTrajectory with specific chunk size
        n_frames = 10
        chunk_size = 3
        expected_chunks = (n_frames + chunk_size - 1) // chunk_size  # ceil(10/3) = 4

        factory, _ = dask_traj_factory
        dask_traj = factory(n_frames=n_frames, n_atoms=80, chunk_size=chunk_size)

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [dask_traj]
        pipeline_data.trajectory_data.trajectory_names = ["chunk_mock_test"]

        # Mock the superpose method to count calls
        with patch.object(md.Trajectory, 'superpose') as mock_superpose:
            # Superpose
            traj_manager.superpose(
                pipeline_data,
                traj_selection=0,
                reference_traj=0,
                reference_frame=0,
                atom_selection="all"
            )

            # Verify superpose was called once per chunk
            assert mock_superpose.call_count == expected_chunks, \
                f"Expected {expected_chunks} superpose calls for {n_frames} frames with chunk_size {chunk_size}, " \
                f"but got {mock_superpose.call_count}"

            # Verify each call had the correct parameters structure
            for call in mock_superpose.call_args_list:
                args, kwargs = call
                assert 'reference' in kwargs or len(args) >= 1, "superpose should be called with reference"
                assert 'atom_indices' in kwargs or len(args) >= 2, "superpose should be called with atom_indices"

    def test_dask_trajectory_persistence_after_superpose(self, dask_traj_factory):
        """Test that alignment persists after reloading DaskMDTrajectory."""
        # Setup
        pipeline_data = PipelineData()
        traj_manager = TrajectoryManager(use_memmap=True)

        # Create DaskMDTrajectory with drift
        factory, register = dask_traj_factory
        dask_traj = factory(n_frames=6, n_atoms=80, add_drift=True, chunk_size=2)

        # Store original file paths for reloading
        trajectory_file = dask_traj.trajectory_file
        topology_file = dask_traj.topology_file
        zarr_cache_path = dask_traj.zarr_cache_path
        chunk_size = dask_traj.chunk_size

        # Setup pipeline data
        pipeline_data.trajectory_data = TrajectoryData()
        pipeline_data.trajectory_data.trajectories = [dask_traj]
        pipeline_data.trajectory_data.trajectory_names = ["persistence_test"]

        # Store original frame count and atoms for later comparison
        original_n_frames = dask_traj.n_frames
        original_n_atoms = dask_traj.n_atoms

        # Superpose
        traj_manager.superpose(
            pipeline_data,
            traj_selection=0,
            reference_traj=0,
            reference_frame=0,
            atom_selection="all"
        )

        # Reload trajectory from Zarr store
        reloaded_traj = DaskMDTrajectory(
            trajectory_file=trajectory_file,
            topology_file=topology_file,
            zarr_cache_path=zarr_cache_path,
            chunk_size=chunk_size
        )
        register(reloaded_traj)

        # Verify basic properties are maintained
        assert reloaded_traj.n_frames == original_n_frames, "Frame count should be preserved"
        assert reloaded_traj.n_atoms == original_n_atoms, "Atom count should be preserved"

        # Verify coordinates are accessible
        assert reloaded_traj.xyz is not None, "Reloaded trajectory should have accessible coordinates"

        # Reference frame should have low RMSD (aligned to itself)
        rmsd_ref = md.rmsd(reloaded_traj, reloaded_traj[0], superpose=False)
        assert rmsd_ref[0] < 1e-6, "Reference frame should have zero RMSD"
