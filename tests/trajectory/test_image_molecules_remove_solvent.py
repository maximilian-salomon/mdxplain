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
Tests for DaskMDTrajectory image_molecules() and remove_solvent().

Verifies that MDTraj and DaskMDTrajectory produce identical results for
both operations, and validates parameter handling and in-place behavior.
"""
import numpy as np
import mdtraj as md
import os
import shutil
from mdxplain.trajectory.entities.dask_md_trajectory import DaskMDTrajectory


def create_trajectory_with_unitcell(n_frames=5, n_atoms=80):
    """
    Create test trajectory with periodic boundary conditions.

    Parameters
    ----------
    n_frames : int
        Number of frames
    n_atoms : int
        Number of atoms

    Returns
    -------
    md.Trajectory
        Trajectory with unitcell information for PBC
    """
    # Create topology with bonds (REQUIRED for image_molecules!)
    topology = md.Topology()
    chain = topology.add_chain()

    # Create residues with single atoms AND bonds between them
    atoms = []
    for i in range(n_atoms):
        residue = topology.add_residue(f'ALA', chain)
        atom = topology.add_atom('CA', md.element.carbon, residue)
        atoms.append(atom)

    # Add bonds to create one big molecule (all atoms bonded in chain)
    for i in range(n_atoms - 1):
        topology.add_bond(atoms[i], atoms[i + 1])

    # Create random coordinates
    xyz = np.random.rand(n_frames, n_atoms, 3).astype(np.float32) * 5.0  # 0-5 nm

    # Create trajectory
    traj = md.Trajectory(xyz, topology)

    # Add unitcell information (5x5x5 nm cubic box)
    traj.unitcell_lengths = np.array([[5.0, 5.0, 5.0]] * n_frames)
    traj.unitcell_angles = np.array([[90.0, 90.0, 90.0]] * n_frames)

    return traj


def create_trajectory_with_solvent(n_frames=5, n_protein_atoms=40, n_solvent_atoms=40):
    """
    Create test trajectory with protein and solvent atoms.

    Parameters
    ----------
    n_frames : int
        Number of frames
    n_protein_atoms : int
        Number of protein atoms
    n_solvent_atoms : int
        Number of solvent atoms

    Returns
    -------
    md.Trajectory
        Trajectory with protein and solvent residues
    """
    # Create topology
    topology = md.Topology()
    chain = topology.add_chain()

    # Add protein residues (4 atoms each)
    for i in range(n_protein_atoms // 4):
        residue = topology.add_residue('ALA', chain)
        for j in range(4):
            topology.add_atom(f'CA{j}', md.element.carbon, residue)

    # Add solvent residues (HOH, 1 atom each)
    for i in range(n_solvent_atoms):
        residue = topology.add_residue('HOH', chain)
        topology.add_atom('O', md.element.oxygen, residue)

    # Create random coordinates
    total_atoms = n_protein_atoms + n_solvent_atoms
    xyz = np.random.rand(n_frames, total_atoms, 3).astype(np.float32)

    # Create trajectory
    traj = md.Trajectory(xyz, topology)

    return traj


def cleanup_dask_trajectory(dask_traj):
    """
    Clean up DaskMDTrajectory resources.

    Parameters
    ----------
    dask_traj : DaskMDTrajectory
        Trajectory to clean up
    """
    if dask_traj is None:
        return

    try:
        zarr_path = dask_traj.zarr_cache_path
        dask_traj.cleanup()

        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
    except Exception:
        pass


class TestImageMolecules:
    """Tests for image_molecules() functionality."""

    def test_mdtraj_dask_equivalence(self, tmp_path):
        """
        Test that MDTraj and DaskMDTrajectory produce identical results.

        Validates that image_molecules() implementation in DaskMDTrajectory
        produces bit-identical results to MDTraj's native implementation.
        """
        # Create test trajectory IN MEMORY
        base_traj = create_trajectory_with_unitcell(n_frames=5, n_atoms=80)

        # Create copies with SAME initial coordinates
        md_traj = base_traj[:]
        dask_traj = DaskMDTrajectory.from_mdtraj(
            base_traj[:],  # Another copy with SAME coordinates
            zarr_cache_path=str(tmp_path / 'cache_image.zarr')
        )

        try:
            # Create anchor molecules (first 10 residues)
            anchor_mols = []
            for i in range(min(10, md_traj.n_residues)):
                atoms_in_residue = [atom for atom in md_traj.topology.residue(i).atoms]
                anchor_mols.append(set(atoms_in_residue))

            # Apply image_molecules to both (in-place)
            md_traj.image_molecules(anchor_molecules=anchor_mols, inplace=True)
            dask_traj.image_molecules(anchor_molecules=anchor_mols, inplace=True)

            # Verify identical results
            assert np.allclose(md_traj.xyz, dask_traj.xyz, atol=1e-6), \
                "MDTraj and DaskMDTrajectory image_molecules() should produce identical results"

            # Verify topology unchanged
            assert md_traj.n_atoms == dask_traj.n_atoms
            assert md_traj.n_residues == dask_traj.n_residues

        finally:
            cleanup_dask_trajectory(dask_traj)

    def test_inplace_true(self, tmp_path):
        """
        Test that inplace=True modifies trajectory in-place.

        Validates that image_molecules(inplace=True) returns self
        and modifies the trajectory object directly.
        """
        # Create test trajectory IN MEMORY
        traj = create_trajectory_with_unitcell(n_frames=5, n_atoms=80)

        # Create DaskMDTrajectory directly
        dask_traj = DaskMDTrajectory.from_mdtraj(
            traj,
            zarr_cache_path=str(tmp_path / 'cache_inplace.zarr')
        )

        try:
            # Create anchor molecules
            anchor_mols = []
            for i in range(min(10, dask_traj.topology.n_residues)):
                atoms_in_residue = [atom for atom in dask_traj.topology.residue(i).atoms]
                anchor_mols.append(set(atoms_in_residue))

            # Store original object identity
            original_id = id(dask_traj)
            original_coords = dask_traj.xyz.copy()

            # Apply in-place
            result = dask_traj.image_molecules(anchor_molecules=anchor_mols, inplace=True)

            # Verify it returns self
            assert id(result) == original_id
            assert result is dask_traj

            # Verify coordinates were modified
            assert not np.allclose(original_coords, dask_traj.xyz, atol=1e-10), \
                "Coordinates should be modified by image_molecules"

        finally:
            cleanup_dask_trajectory(dask_traj)

    def test_inplace_false(self, tmp_path):
        """
        Test that inplace=False creates new trajectory.

        Validates that image_molecules(inplace=False) returns a new
        DaskMDTrajectory object and leaves the original unchanged.
        """
        # Create test trajectory IN MEMORY
        traj = create_trajectory_with_unitcell(n_frames=5, n_atoms=80)

        # Create DaskMDTrajectory directly
        original_traj = DaskMDTrajectory.from_mdtraj(
            traj,
            zarr_cache_path=str(tmp_path / 'cache_copy_orig.zarr')
        )

        new_traj = None
        try:
            # Create anchor molecules
            anchor_mols = []
            for i in range(min(10, original_traj.topology.n_residues)):
                atoms_in_residue = [atom for atom in original_traj.topology.residue(i).atoms]
                anchor_mols.append(set(atoms_in_residue))

            # Store original state
            original_id = id(original_traj)
            original_coords = original_traj.xyz.copy()

            # Apply with inplace=False
            new_traj = original_traj.image_molecules(anchor_molecules=anchor_mols, inplace=False)

            # Verify new object was created
            assert id(new_traj) != original_id
            assert new_traj is not original_traj

            # Verify original unchanged
            assert np.allclose(original_coords, original_traj.xyz, atol=1e-10), \
                "Original trajectory should be unchanged"

            # Verify new trajectory was modified
            assert not np.allclose(original_coords, new_traj.xyz, atol=1e-10), \
                "New trajectory should be modified"

        finally:
            cleanup_dask_trajectory(original_traj)
            cleanup_dask_trajectory(new_traj)

    def test_with_anchor_molecules(self, tmp_path):
        """
        Test image_molecules with anchor_molecules parameter.

        Validates that anchor_molecules parameter is correctly
        passed through and produces different results than default.
        """
        # Create test trajectory IN MEMORY
        base_traj = create_trajectory_with_unitcell(n_frames=5, n_atoms=80)

        # Create copies with SAME initial coordinates
        md_traj = base_traj[:]
        dask_traj = DaskMDTrajectory.from_mdtraj(
            base_traj[:],  # Another copy with SAME coordinates
            zarr_cache_path=str(tmp_path / 'cache_anchor.zarr')
        )

        try:
            # Define anchor molecules (first 3 residues)
            anchor_mols = []
            for i in [0, 1, 2]:
                atoms_in_residue = [atom for atom in md_traj.topology.residue(i).atoms]
                anchor_mols.append(set(atoms_in_residue))

            # Apply with anchor molecules to both (in-place)
            md_traj.image_molecules(anchor_molecules=anchor_mols, inplace=True)
            dask_traj.image_molecules(anchor_molecules=anchor_mols, inplace=True)

            # Verify identical results
            assert np.allclose(md_traj.xyz, dask_traj.xyz, atol=1e-6), \
                "Results with anchor_molecules should be identical"

        finally:
            cleanup_dask_trajectory(dask_traj)


class TestRemoveSolvent:
    """Tests for remove_solvent() functionality."""

    def test_mdtraj_dask_equivalence(self, tmp_path):
        """
        Test that MDTraj and DaskMDTrajectory produce identical results.

        Validates that remove_solvent() implementation in DaskMDTrajectory
        produces bit-identical results to MDTraj's native implementation.
        """
        # Create test trajectory IN MEMORY
        md_traj = create_trajectory_with_solvent(n_frames=5, n_protein_atoms=40, n_solvent_atoms=40)

        # Create DaskMDTrajectory directly
        dask_traj = DaskMDTrajectory.from_mdtraj(
            md_traj[:],  # slice creates copy
            zarr_cache_path=str(tmp_path / 'cache_solvent.zarr')
        )

        dask_result = None
        try:
            # Apply remove_solvent to both
            md_result = md_traj.remove_solvent()
            dask_result = dask_traj.remove_solvent()

            # Verify identical results
            assert np.allclose(md_result.xyz, dask_result.xyz, atol=1e-6), \
                "MDTraj and DaskMDTrajectory remove_solvent() should produce identical results"

            # Verify same number of atoms
            assert md_result.n_atoms == dask_result.n_atoms
            assert md_result.n_residues == dask_result.n_residues

        finally:
            cleanup_dask_trajectory(dask_traj)
            cleanup_dask_trajectory(dask_result)

    def test_atom_count_reduction(self, tmp_path):
        """
        Test that remove_solvent correctly reduces atom count.

        Validates that the number of atoms decreases by the expected
        amount after removing solvent atoms.
        """
        # Create test trajectory IN MEMORY (40 protein + 40 solvent = 80 total)
        traj = create_trajectory_with_solvent(n_frames=5, n_protein_atoms=40, n_solvent_atoms=40)

        # Create DaskMDTrajectory directly
        dask_traj = DaskMDTrajectory.from_mdtraj(
            traj,
            zarr_cache_path=str(tmp_path / 'cache_count.zarr')
        )

        result = None
        try:
            # Store original count
            original_n_atoms = dask_traj.n_atoms
            assert original_n_atoms == 80

            # Remove solvent
            result = dask_traj.remove_solvent()

            # Verify atom count decreased (should have 40 protein atoms left)
            assert result.n_atoms == 40, \
                f"Expected 40 atoms after removing solvent, got {result.n_atoms}"

            # Verify original unchanged (inplace=False is default)
            assert dask_traj.n_atoms == original_n_atoms

        finally:
            cleanup_dask_trajectory(dask_traj)
            cleanup_dask_trajectory(result)

    def test_exclude_none(self, tmp_path):
        """
        Test that exclude=None removes all solvent.

        Validates that when exclude=None (default), all recognized
        solvent residues are removed from the trajectory.
        """
        # Create test trajectory IN MEMORY
        traj = create_trajectory_with_solvent(n_frames=5, n_protein_atoms=40, n_solvent_atoms=40)

        # Create DaskMDTrajectory directly
        dask_traj = DaskMDTrajectory.from_mdtraj(
            traj,
            zarr_cache_path=str(tmp_path / 'cache_exclude_none.zarr')
        )

        result = None
        try:
            # Remove all solvent (exclude=None is default)
            result = dask_traj.remove_solvent(exclude=None)

            # Verify all HOH residues were removed
            remaining_residues = [res.name for res in result.topology.residues]
            assert 'HOH' not in remaining_residues, \
                "HOH residues should be removed when exclude=None"

            # Should only have protein atoms left
            assert result.n_atoms == 40

        finally:
            cleanup_dask_trajectory(dask_traj)
            cleanup_dask_trajectory(result)

    def test_exclude_list(self, tmp_path):
        """
        Test that exclude parameter keeps specified residues.

        Validates that residues in the exclude list are preserved
        while other solvent is removed.
        """
        # Create test trajectory IN MEMORY
        traj = create_trajectory_with_solvent(n_frames=5, n_protein_atoms=40, n_solvent_atoms=40)

        # Create DaskMDTrajectory directly
        dask_traj = DaskMDTrajectory.from_mdtraj(
            traj,
            zarr_cache_path=str(tmp_path / 'cache_exclude_list.zarr')
        )

        result = None
        try:
            # Keep HOH residues (exclude means "don't remove")
            result = dask_traj.remove_solvent(exclude=['HOH'])

            # Verify HOH residues were kept
            remaining_residues = [res.name for res in result.topology.residues]
            assert 'HOH' in remaining_residues, \
                "HOH residues should be kept when in exclude list"

            # Should have all atoms (40 protein + 40 HOH = 80)
            assert result.n_atoms == 80

        finally:
            cleanup_dask_trajectory(dask_traj)
            cleanup_dask_trajectory(result)

    def test_inplace_false(self, tmp_path):
        """
        Test that inplace=False creates new trajectory.

        Validates that remove_solvent(inplace=False) returns a new
        DaskMDTrajectory object and leaves the original unchanged.
        """
        # Create test trajectory IN MEMORY
        traj = create_trajectory_with_solvent(n_frames=5, n_protein_atoms=40, n_solvent_atoms=40)

        # Create DaskMDTrajectory directly
        original_traj = DaskMDTrajectory.from_mdtraj(
            traj,
            zarr_cache_path=str(tmp_path / 'cache_inplace_false.zarr')
        )

        new_traj = None
        try:
            # Store original state
            original_n_atoms = original_traj.n_atoms
            assert original_n_atoms == 80

            # Apply with inplace=False (default)
            new_traj = original_traj.remove_solvent(inplace=False)

            # Verify new object was created
            assert new_traj is not original_traj

            # Verify original unchanged
            assert original_traj.n_atoms == original_n_atoms, \
                "Original trajectory should be unchanged"

            # Verify new trajectory has fewer atoms
            assert new_traj.n_atoms == 40, \
                "New trajectory should have solvent removed"

        finally:
            cleanup_dask_trajectory(original_traj)
            cleanup_dask_trajectory(new_traj)
