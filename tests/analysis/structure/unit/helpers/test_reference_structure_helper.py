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

"""Tests for ReferenceStructureHelper."""

import mdtraj as md
import numpy as np
import pytest

from mdxplain.analysis.structure.helpers.reference_structure_helper import ReferenceStructureHelper


class TestReferenceStructureHelper:
    """Test class for ReferenceStructureHelper."""

    def create_simple_topology(self, n_atoms=2):
        """Create simple topology for testing.

        Parameters
        ----------
        n_atoms : int, optional
            Number of atoms to create. Defaults to 2.

        Returns
        -------
        md.Topology
            Simple topology with carbon atoms.
        """
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain)
        for i in range(n_atoms):
            topology.add_atom(f"C{i}", md.element.carbon, residue)
        return topology

    def create_test_trajectories(self):
        """Create test trajectories with known mean and median values.

        Returns
        -------
        tuple
            Two trajectories with calculable reference coordinates.
        """
        topology = self.create_simple_topology(n_atoms=2)

        # Trajectory 1: Y-values 0, 1 for frames
        coords1 = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Frame 0
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],  # Frame 1
        ], dtype=np.float32)

        # Trajectory 2: Y-values 2, 3 for frames
        coords2 = np.array([
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],  # Frame 0
            [[0.0, 3.0, 0.0], [1.0, 3.0, 0.0]],  # Frame 1
        ], dtype=np.float32)

        traj1 = md.Trajectory(coords1, topology)
        traj2 = md.Trajectory(coords2, topology)

        return traj1, traj2

    def create_large_trajectory(self, n_frames=10):
        """Create larger trajectory for memmap testing.

        Parameters
        ----------
        n_frames : int, optional
            Number of frames to create. Defaults to 10.

        Returns
        -------
        md.Trajectory
            Trajectory with incremental Y coordinates.
        """
        topology = self.create_simple_topology(n_atoms=3)
        coords = np.zeros((n_frames, 3, 3), dtype=np.float32)

        for frame in range(n_frames):
            for atom in range(3):
                coords[frame, atom] = [atom, frame, 0.0]

        return md.Trajectory(coords, topology)

    def test_mean_coordinates_simple(self):
        """Test get_mean_coordinates with simple cross-trajectory calculation."""
        traj1, traj2 = self.create_test_trajectories()

        result = ReferenceStructureHelper.get_mean_coordinates(
            trajectories=[traj1, traj2],
            atom_chunk_size=10,
            use_memmap=False,
            cross_trajectory=True
        )

        # Expected: mean of Y-values [0,1,2,3] = 1.5 for both atoms
        expected = np.array([[0.0, 1.5, 0.0], [1.0, 1.5, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        assert result.shape == (2, 3)

    def test_mean_coordinates_per_trajectory(self):
        """Test get_mean_coordinates with cross_trajectory=False."""
        traj1, traj2 = self.create_test_trajectories()

        result = ReferenceStructureHelper.get_mean_coordinates(
            trajectories=[traj1, traj2],
            atom_chunk_size=10,
            use_memmap=False,
            cross_trajectory=False
        )

        # Expected: traj1 mean Y = 0.5, traj2 mean Y = 2.5
        expected_traj1 = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)
        expected_traj2 = np.array([[0.0, 2.5, 0.0], [1.0, 2.5, 0.0]], dtype=np.float32)

        assert isinstance(result, dict)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], expected_traj1, decimal=6)
        np.testing.assert_array_almost_equal(result[1], expected_traj2, decimal=6)

    def test_mean_coordinates_single_trajectory(self):
        """Test get_mean_coordinates with single trajectory."""
        traj1, _ = self.create_test_trajectories()

        result = ReferenceStructureHelper.get_mean_coordinates(
            trajectories=[traj1],
            atom_chunk_size=10,
            use_memmap=False,
            cross_trajectory=True
        )

        # Expected: mean of Y-values [0,1] = 0.5 for both atoms
        expected = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_mean_coordinates_with_memmap(self):
        """Test get_mean_coordinates with use_memmap=True."""
        large_traj = self.create_large_trajectory(n_frames=8)

        result = ReferenceStructureHelper.get_mean_coordinates(
            trajectories=[large_traj],
            atom_chunk_size=10,
            use_memmap=True,
            cross_trajectory=True,
            frame_chunk_size=3
        )

        # Expected: mean of Y-values [0,1,2,3,4,5,6,7] = 3.5 for all atoms
        expected = np.array([[0.0, 3.5, 0.0], [1.0, 3.5, 0.0], [2.0, 3.5, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_mean_coordinates_empty_trajectories_error(self):
        """Test get_mean_coordinates with empty trajectory list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute mean coordinates without trajectories."):
            ReferenceStructureHelper.get_mean_coordinates(
                trajectories=[],
                atom_chunk_size=10,
                use_memmap=False
            )

    def test_median_coordinates_simple(self):
        """Test get_median_coordinates with simple cross-trajectory calculation."""
        traj1, traj2 = self.create_test_trajectories()

        result = ReferenceStructureHelper.get_median_coordinates(
            trajectories=[traj1, traj2],
            atom_chunk_size=10,
            use_memmap=False,
            cross_trajectory=True
        )

        # Expected: median of Y-values [0,1,2,3] = 1.5 for both atoms
        expected = np.array([[0.0, 1.5, 0.0], [1.0, 1.5, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_median_coordinates_per_trajectory(self):
        """Test get_median_coordinates with cross_trajectory=False."""
        traj1, traj2 = self.create_test_trajectories()

        result = ReferenceStructureHelper.get_median_coordinates(
            trajectories=[traj1, traj2],
            atom_chunk_size=10,
            use_memmap=False,
            cross_trajectory=False
        )

        # Expected: traj1 median Y = 0.5, traj2 median Y = 2.5
        expected_traj1 = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)
        expected_traj2 = np.array([[0.0, 2.5, 0.0], [1.0, 2.5, 0.0]], dtype=np.float32)

        assert isinstance(result, dict)
        np.testing.assert_array_almost_equal(result[0], expected_traj1, decimal=6)
        np.testing.assert_array_almost_equal(result[1], expected_traj2, decimal=6)

    def test_median_coordinates_single_trajectory(self):
        """Test get_median_coordinates with single trajectory."""
        traj1, _ = self.create_test_trajectories()

        result = ReferenceStructureHelper.get_median_coordinates(
            trajectories=[traj1],
            atom_chunk_size=10,
            use_memmap=False,
            cross_trajectory=True
        )

        # Expected: median of Y-values [0,1] = 0.5 for both atoms
        expected = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_median_coordinates_with_memmap(self):
        """Test get_median_coordinates with use_memmap=True."""
        large_traj = self.create_large_trajectory(n_frames=6)

        result = ReferenceStructureHelper.get_median_coordinates(
            trajectories=[large_traj],
            atom_chunk_size=10,
            use_memmap=True,
            cross_trajectory=True,
            frame_chunk_size=2
        )

        # Expected: median of Y-values [0,1,2,3,4,5] = 2.5 for all atoms
        expected = np.array([[0.0, 2.5, 0.0], [1.0, 2.5, 0.0], [2.0, 2.5, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_median_coordinates_empty_trajectories_error(self):
        """Test get_median_coordinates with empty trajectory list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute median coordinates without trajectories."):
            ReferenceStructureHelper.get_median_coordinates(
                trajectories=[],
                atom_chunk_size=10,
                use_memmap=False
            )

    def test_cross_trajectory_atom_mismatch_error(self):
        """Test error when trajectories have different numbers of atoms."""
        topology1 = self.create_simple_topology(n_atoms=2)
        topology2 = self.create_simple_topology(n_atoms=3)  # Different number of atoms

        coords1 = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
        coords2 = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)

        traj1 = md.Trajectory(coords1, topology1)
        traj2 = md.Trajectory(coords2, topology2)

        with pytest.raises(ValueError, match="All trajectories must have the same number of atoms."):
            ReferenceStructureHelper.get_mean_coordinates(
                trajectories=[traj1, traj2],
                atom_chunk_size=10,
                use_memmap=False,
                cross_trajectory=True
            )
