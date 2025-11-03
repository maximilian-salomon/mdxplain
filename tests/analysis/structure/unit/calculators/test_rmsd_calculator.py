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

"""Tests for RMSDCalculator."""

import mdtraj as md
import numpy as np
import pytest
from unittest.mock import patch

from mdxplain.analysis.structure.calculators.rmsd_calculator import RMSDCalculator


class TestRMSDCalculator:
    """Test class for RMSDCalculator."""

    def setup_method(self):
        """Create test trajectories with known RMSD values."""
        # Simple topology with 2 atoms
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain)
        topology.add_atom("CA", md.element.carbon, residue)
        topology.add_atom("CB", md.element.carbon, residue)

        # Reference trajectory (single frame)
        self.ref_coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
        self.ref_traj = md.Trajectory(self.ref_coords, topology)

        # Test trajectory with known displacements
        self.coords = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Frame 0: identical, RMSD = 0.0
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],  # Frame 1: 0.1 shift, RMSD = 0.1
            [[0.0, 0.2, 0.0], [1.0, 0.2, 0.0]],  # Frame 2: 0.2 shift, RMSD = 0.2
            [[0.3, 0.0, 0.0], [1.3, 0.0, 0.0]],  # Frame 3: 0.3 shift, RMSD = 0.3
            [[0.0, 0.0, 0.4], [1.0, 0.0, 0.4]],  # Frame 4: 0.4 shift, RMSD = 0.4
        ], dtype=np.float32)
        self.traj = md.Trajectory(self.coords, topology)

        # Second trajectory for multi-trajectory tests
        self.coords2 = np.array([
            [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],  # All frames shifted by 0.5
            [[0.6, 0.0, 0.0], [1.6, 0.0, 0.0]],  # RMSD = 0.6
            [[0.7, 0.0, 0.0], [1.7, 0.0, 0.0]],  # RMSD = 0.7
        ], dtype=np.float32)
        self.traj2 = md.Trajectory(self.coords2, topology)

        # Single atom trajectory for edge cases
        topology_1atom = md.Topology()
        chain = topology_1atom.add_chain()
        residue = topology_1atom.add_residue("ALA", chain)
        topology_1atom.add_atom("CA", md.element.carbon, residue)

        coords_1atom = np.array([
            [[0.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0]],
        ], dtype=np.float32)
        self.traj_1atom = md.Trajectory(coords_1atom, topology_1atom)

    def make_unique_trajectory(self, n_frames: int = 10) -> md.Trajectory:
        """Create trajectory where frame i has coordinates [i, i, i].

        Parameters
        ----------
        n_frames : int
            Number of frames to create

        Returns
        -------
        md.Trajectory
            Trajectory with unique coordinates per frame for exact testing
        """
        topology = md.Topology()
        chain = topology.add_chain()
        residue = topology.add_residue("ALA", chain)
        topology.add_atom("CA", md.element.carbon, residue)

        coords = np.array([[[float(i), float(i), float(i)]] for i in range(n_frames)], dtype=np.float32)
        return md.Trajectory(coords, topology)

    # 1. Konstruktor Tests (6 Tests)

    def test_init_valid_single_trajectory(self):
        """

        Test RMSDCalculator initialization with single trajectory.

        Verifies basic constructor functionality with minimal parameters.
        """
        calc = RMSDCalculator([self.traj], chunk_size=10, use_memmap=False)
        assert len(calc.trajectories) == 1
        assert calc.trajectories[0] is self.traj
        assert calc.chunk_size == 10
        assert calc.use_memmap is False

    def test_init_valid_multiple_trajectories(self):
        """

        Test RMSDCalculator initialization with multiple trajectories.

        Verifies constructor handles multiple trajectory objects correctly.
        """
        calc = RMSDCalculator([self.traj, self.traj2], chunk_size=5, use_memmap=False)
        assert len(calc.trajectories) == 2
        assert calc.trajectories[0] is self.traj
        assert calc.trajectories[1] is self.traj2

    def test_init_empty_trajectories_error(self):
        """

        Test RMSDCalculator initialization with empty trajectory list.

        Should raise ValueError when no trajectories are provided.
        """
        with pytest.raises(ValueError, match="RMSDCalculator requires at least one trajectory"):
            RMSDCalculator([], chunk_size=10, use_memmap=False)

    def test_init_with_memmap_true(self):
        """

        Test RMSDCalculator initialization with memmap enabled.

        Verifies memmap flag is correctly stored and accessible.
        """
        calc = RMSDCalculator([self.traj], chunk_size=2, use_memmap=True)
        assert calc.use_memmap is True
        assert calc.chunk_size == 2

    def test_init_with_memmap_false(self):
        """

        Test RMSDCalculator initialization with memmap disabled.

        Verifies memmap flag is correctly set to False.
        """
        calc = RMSDCalculator([self.traj], chunk_size=100, use_memmap=False)
        assert calc.use_memmap is False

    def test_init_different_chunk_sizes(self):
        """

        Test RMSDCalculator initialization with different chunk sizes.

        Verifies chunk_size parameter is correctly stored for different values.
        """
        calc1 = RMSDCalculator([self.traj], chunk_size=1, use_memmap=False)
        calc2 = RMSDCalculator([self.traj], chunk_size=1000, use_memmap=False)
        assert calc1.chunk_size == 1
        assert calc2.chunk_size == 1000

    # 2. rmsd_to_reference Tests (12 Tests)

    def test_rmsd_to_reference_identical_frames(self):
        """Test RMSD to reference with known coordinate shifts.

        Expected RMSD values for uniform shifts:

        - Frame 0: 0.0 (identical to reference)
        - Frame 1: 0.1 (0.1 shift in X for both atoms)
        - Frame 2: 0.2 (0.2 shift in Y for both atoms)
        - Frame 3: 0.3 (0.3 shift in X for both atoms)
        - Frame 4: 0.4 (0.4 shift in Z for both atoms)
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected = [np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rmsd_to_reference_known_shift(self):
        """Test RMSD to reference with second trajectory.

        Second trajectory has uniform 0.5 base shift plus increments.
        Expected RMSD values: [0.5, 0.6, 0.7]
        """
        calc = RMSDCalculator([self.traj2], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected = [np.array([0.5, 0.6, 0.7], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rmsd_to_reference_multiple_trajectories(self):
        """Test RMSD to reference with multiple trajectories.

        Expected RMSD values:

        - Trajectory 1: [0.0, 0.1, 0.2, 0.3, 0.4] (uniform shifts)
        - Trajectory 2: [0.5, 0.6, 0.7] (uniform shifts from 0.5)
        """
        calc = RMSDCalculator([self.traj, self.traj2], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        expected2 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        expected = [expected1, expected2]
        # Test structure and individual arrays
        assert len(result) == len(expected)
        for actual, exp in zip(result, expected):
            np.testing.assert_array_almost_equal(actual, exp, decimal=5)

    def test_rmsd_to_reference_metric_mean(self):
        """Test RMSD to reference with mean metric.

        Mean metric averages squared distances per atom.
        Expected RMSD values: [0.0, 0.1, 0.2, 0.3, 0.4]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected = [np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rmsd_to_reference_metric_median(self):
        """Test RMSD to reference with median metric.

        Trajectory with different per-atom shifts:
        
        - Frame 0: [0.0,0.0,0.0], [1.0,0.0,0.0] → RMSD = 0.0
        - Frame 1: [0.1,0.0,0.0], [1.2,0.0,0.0] → squared distances [0.01, 0.04], median = 0.025, RMSD = 0.158
        """
        coords = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ], dtype=np.float32)
        traj = md.Trajectory(coords, self.traj.topology)
        calc = RMSDCalculator([traj], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "median")
        expected = [np.array([0.0, 0.158], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_rmsd_to_reference_metric_mad(self):
        """Test RMSD to reference with MAD metric.

        Create trajectory with different per-atom movements for meaningful MAD values.
        MAD measures variability between atoms - non-zero when atoms move differently.
        """
        coords_mad = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Frame 0: reference
            [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]],  # Frame 1: Atom 0 +0.1, Atom 1 +0.3
            [[0.0, 0.2, 0.0], [1.0, 0.4, 0.0]],  # Frame 2: Atom 0 +0.2Y, Atom 1 +0.4Y
        ], dtype=np.float32)
        traj_mad = md.Trajectory(coords_mad, self.traj.topology)
        calc = RMSDCalculator([traj_mad], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mad")
        expected = [np.array([0.0, 0.1, 0.1], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_rmsd_to_reference_single_atom(self):
        """Test RMSD to reference with single atom selection.

        Using only atom 0, expected RMSD values: [0.0, 0.1, 0.2, 0.3, 0.4]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, [0], "mean")
        expected = [np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rmsd_to_reference_atom_subset(self):
        """Test RMSD to reference with different atom subset.

        Using only atom 1, expected RMSD values: [0.0, 0.1, 0.2, 0.3, 0.4]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, [1], "mean")
        expected = [np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rmsd_to_reference_all_atoms(self):
        """Test RMSD to reference with all atoms (None selection).

        Using all atoms (None), expected RMSD values: [0.0, 0.1, 0.2, 0.3, 0.4]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected = [np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rmsd_to_reference_with_memmap(self):
        """Test RMSD to reference with memmap enabled.

        Memmap = True, chunk size = 2, expected RMSD values: [0.0, 0.1, 0.2, 0.3, 0.4]
        """
        calc = RMSDCalculator([self.traj], chunk_size=2, use_memmap=True)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected = [np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rmsd_to_reference_no_chunking(self):
        """

        Test RMSD to reference with large chunk size (no chunking).

        Chunk size larger than trajectory processes all frames at once.
        """
        calc = RMSDCalculator([self.traj], chunk_size=100, use_memmap=True)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected = [np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    # 3. frame_to_frame Tests (11 Tests)

    def test_frame_to_frame_lag_1(self):
        """

        Test frame-to-frame RMSD calculation with lag=1.

        Expected RMSD values for consecutive frame comparisons.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.frame_to_frame(None, lag=1, metric="mean")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frame_to_frame_lag_2(self):
        """

        Test frame-to-frame RMSD calculation with lag=2.

        Expected RMSD values for frame comparisons with 2-frame spacing.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.frame_to_frame(None, lag=2, metric="mean")
        expected = [np.array([0.2, 0.2, 0.447], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frame_to_frame_lag_3(self):
        """

        Test frame-to-frame RMSD calculation with lag=3.

        Expected RMSD values for frame comparisons with 3-frame spacing.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.frame_to_frame(None, lag=3, metric="mean")
        expected = [np.array([0.3, 0.412], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frame_to_frame_lag_exceeds_trajectory(self):
        """

        Test frame-to-frame RMSD with lag exceeding trajectory length.

        Should return empty array when lag is larger than available frames.
        Also expects RuntimeWarning about trajectory being shorter than requested lag.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.warns(RuntimeWarning, match="Trajectory shorter than the requested lag"):
            result = calc.frame_to_frame(None, lag=10, metric="mean")
        expected = [np.array([], dtype=np.float32)]
        np.testing.assert_array_equal(result, expected)

    def test_frame_to_frame_metric_mean(self):
        """Test frame-to-frame RMSD with mean metric.

        For lag=1 with coordinate shifts [0.0, 0.1, 0.2, 0.3, 0.4]:
        Expected frame-to-frame RMSD: [0.1, 0.224, 0.361, 0.5]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.frame_to_frame(None, lag=1, metric="mean")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frame_to_frame_metric_median(self):
        """

        Test frame-to-frame RMSD with median metric.

        Uses median of squared distances instead of mean for RMSD calculation.
        """
        coords = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.3, 0.0, 0.0]],
        ], dtype=np.float32)
        traj = md.Trajectory(coords, self.traj.topology)
        calc = RMSDCalculator([traj], 10, False)
        result = calc.frame_to_frame(None, lag=1, metric="median")
        # Frame 0→1: median([0.01, 0.04])=0.025, sqrt=0.158
        # Frame 1→2: median([0.01, 0.01])=0.01, sqrt=0.1
        expected = [np.array([0.158, 0.1], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frame_to_frame_metric_mad(self):
        """Test frame-to-frame RMSD with MAD metric.

        Using trajectory with different per-atom movements for meaningful MAD.
        """
        coords_mad = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Frame 0
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],  # Frame 1: different shifts
            [[0.0, 0.1, 0.0], [1.0, 0.3, 0.0]],  # Frame 2: different shifts
            [[0.2, 0.0, 0.0], [1.1, 0.0, 0.0]],  # Frame 3: different shifts
        ], dtype=np.float32)
        traj_mad = md.Trajectory(coords_mad, self.traj.topology)
        calc = RMSDCalculator([traj_mad], 10, False)
        result = calc.frame_to_frame(None, lag=1, metric="mad")
        expected = [np.array([0.05, 0.11, 0.046], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frame_to_frame_with_atom_selection(self):
        """Test frame-to-frame RMSD with atom selection.

        Using only atom 0 with lag=1, expected RMSD: [0.1, 0.224, 0.361, 0.5]
        Atom 0 moves in different directions with increasing distances.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.frame_to_frame([0], lag=1, metric="mean")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frame_to_frame_without_atom_selection(self):
        """Test frame-to-frame RMSD without atom selection.

        Expected RMSD values for lag=1: [0.1, 0.224, 0.361, 0.5]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.frame_to_frame(None, lag=1, metric="mean")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_frame_to_frame_with_chunking(self):
        """
        Test frame-to-frame RMSD with chunking enabled.

        Chunk size=2, lag=1, expected RMSD values: [0.1, 0.224, 0.361, 0.5]
        """
        calc = RMSDCalculator([self.traj], chunk_size=2, use_memmap=True)
        result = calc.frame_to_frame(None, lag=1, metric="mean")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    # 4. window frame_to_start Tests (11 Tests)

    def test_window_frame_to_start_size_2_stride_1(self):
        """
        Test window RMSD frame_to_start with size=2, stride=1.

        Expected RMSD values: [0.1, 0.224, 0.361, 0.5]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, window_size=2, stride=1, metric="mean", mode="frame_to_start")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_size_3_stride_3(self):
        """
        Test window RMSD frame_to_start with size=3, stride=3.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, window_size=3, stride=3, metric="mean", mode="frame_to_start")
        expected = [np.array([0.15], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_size_2_stride_2(self):
        """
        Test window RMSD frame_to_start with size=2, stride=2.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, window_size=2, stride=2, metric="mean", mode="frame_to_start")
        expected = [np.array([0.1, 0.361], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_size_4_stride_1(self):
        """
        Test window RMSD frame_to_start with size=4, stride=1.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, window_size=4, stride=1, metric="mean", mode="frame_to_start")
        expected = [np.array([0.2, 0.279], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_metric_mean(self):
        """Test window RMSD frame_to_start with mean metric.

        Window size=2, step=1, comparing each frame to first frame in window.
        Expected RMSD values: [0.1, 0.224, 0.361, 0.5]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 2, 1, "mean", "frame_to_start")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_metric_median(self):
        """Test window RMSD frame_to_start with median metric.

        Window size=3, step=1, frames compared to first in each window.
        Expected RMSD values: [0.15, 0.25, 0.35] (median of comparisons)
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 3, 1, "median", "frame_to_start")
        expected = [np.array([0.15, 0.212, 0.404], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_metric_mad(self):
        """Test window RMSD frame_to_start with MAD metric.

        Use trajectory with different per-atom movements and window size=3 for meaningful MAD.
        """
        coords_mad = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Frame 0
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],  # Frame 1: different shifts
            [[0.0, 0.2, 0.0], [1.0, 0.4, 0.0]],  # Frame 2: different shifts
            [[0.3, 0.0, 0.0], [1.1, 0.0, 0.0]],  # Frame 3: different shifts
            [[0.0, 0.0, 0.4], [1.0, 0.0, 0.5]],  # Frame 4: different shifts
        ], dtype=np.float32)
        traj_mad = md.Trajectory(coords_mad, self.traj.topology)
        calc = RMSDCalculator([traj_mad], 10, False)
        result = calc.window(None, 3, 1, "mad", "frame_to_start")
        expected = [np.array([0.025, 0.031, 0.035], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_with_atoms(self):
        """Test window RMSD frame_to_start with atom selection.

        Window size=2, stride=1, atom_selection=[0], comparing to first frame in each window.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window([0], 2, 1, "mean", "frame_to_start")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_without_atoms(self):
        """Test window frame_to_start without atom selection.

        Window size=2, stride=1, expected RMSD values: [0.1, 0.224, 0.361, 0.5]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 2, 1, "mean", "frame_to_start")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_start_chunking(self):
        """Test window RMSD frame_to_start with chunking enabled.

        Window size=2, stride=1, chunk_size=2, comparing to first frame in each window.
        """
        calc = RMSDCalculator([self.traj], chunk_size=2, use_memmap=True)
        result = calc.window(None, 2, 1, "mean", "frame_to_start")
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    # 5. window frame_to_frame Tests (12 Tests)

    def test_window_frame_to_frame_lag_1_size_3_stride_1(self):
        """Test window RMSD frame_to_frame with lag=1, size=3, stride=1.

        Window size=3, stride=1, lag=1, comparing adjacent frames within windows.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 3, 1, "mean", "frame_to_frame", lag=1)
        expected = [np.array([0.162, 0.292, 0.430], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_lag_2_size_4_stride_2(self):
        """Test window RMSD frame_to_frame with lag=2, size=4, stride=2.

        Window size=4, stride=2, lag=2, comparing frames with gap of 2 within windows.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 4, 2, "mean", "frame_to_frame", lag=2)
        expected = [np.array([0.2], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_lag_1_size_2_stride_1(self):
        """Test window RMSD frame_to_frame with lag=1, size=2, stride=1.

        Window size=2, stride=1, lag=1, comparing adjacent frames within windows.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 2, 1, "mean", "frame_to_frame", lag=1)
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_lag_3_size_5_stride_3(self):
        """Test window RMSD frame_to_frame with lag=3, size=5, stride=3.

        Window size=5, stride=3, lag=3, comparing frames with gap of 3 within windows.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 5, 3, "mean", "frame_to_frame", lag=3)
        expected = [np.array([0.356], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_metric_mean(self):
        """Test window RMSD frame_to_frame with mean metric.

        Window size=2, step=1, lag=1, comparing adjacent frames within windows.
        Expected RMSD values: [0.1, 0.224, 0.361, 0.5]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 2, 1, "mean", "frame_to_frame", lag=1)
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_metric_median(self):
        """Test window RMSD frame_to_frame with median metric.

        Window size=3, step=1, lag=1, median of frame-to-frame comparisons.
        Expected RMSD values: [0.162, 0.292, 0.430] (median of multiple comparisons)
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 3, 1, "median", "frame_to_frame", lag=1)
        expected = [np.array([0.162, 0.292, 0.430], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_metric_mad(self):
        """Test window RMSD frame_to_frame with MAD metric.

        Use trajectory with different per-atom movements for meaningful MAD values.
        """
        coords_mad = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # Frame 0
            [[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]],  # Frame 1: different shifts
            [[0.0, 0.2, 0.0], [1.0, 0.4, 0.0]],  # Frame 2: different shifts
            [[0.3, 0.0, 0.0], [1.1, 0.0, 0.0]],  # Frame 3: different shifts
            [[0.0, 0.0, 0.4], [1.0, 0.0, 0.5]],  # Frame 4: different shifts
        ], dtype=np.float32)
        traj_mad = md.Trajectory(coords_mad, self.traj.topology)
        calc = RMSDCalculator([traj_mad], 10, False)
        result = calc.window(None, 3, 1, "mad", "frame_to_frame", lag=1)
        expected = [np.array([0.031, 0.043, 0.01], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_with_atoms(self):
        """Test window RMSD frame_to_frame with atom selection.

        Window size=2, stride=1, lag=1, atom_selection=[0], comparing adjacent frames.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window([0], 2, 1, "mean", "frame_to_frame", lag=1)
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_without_atoms(self):
        """Test window frame_to_frame without atom selection.

        Window size=2, stride=1, lag=1, expected RMSD: [0.1, 0.224, 0.361, 0.5]
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 2, 1, "mean", "frame_to_frame", lag=1)
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_frame_to_frame_without_lag_error(self):
        """Test window RMSD frame_to_frame error when lag not provided.

        Expected ValueError for missing lag parameter in frame_to_frame mode.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="lag must be provided for frame_to_frame mode"):
            calc.window(None, 2, 1, "mean", "frame_to_frame", lag=None)

    def test_window_frame_to_frame_negative_lag_error(self):
        """Test window RMSD frame_to_frame error for negative lag.

        Expected ValueError for negative lag values in frame_to_frame mode.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="lag must be positive"):
            calc.window(None, 2, 1, "mean", "frame_to_frame", lag=-1)

    def test_window_invalid_mode_error(self):
        """Test window RMSD error for invalid mode.

        Expected ValueError for unsupported mode parameter values.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="mode must be 'frame_to_start' or 'frame_to_frame'"):
            calc.window(None, 2, 1, "mean", mode="invalid")

    # 6. Edge Cases (15 Tests)

    def test_single_frame_trajectory(self):
        """Test RMSD with single-frame trajectory.

        Single frame against reference should return [0.0] RMSD.
        """
        single = md.Trajectory(self.coords[:1], self.traj.topology)
        calc = RMSDCalculator([single], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected = [np.array([0.0], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_two_frame_trajectory(self):
        """Test RMSD with two-frame trajectory.

        Two frames should compute proper RMSD values against reference.
        """
        two = md.Trajectory(self.coords[:2], self.traj.topology)
        calc = RMSDCalculator([two], 10, False)
        result = calc.frame_to_frame(None, lag=1, metric="mean")
        expected = [np.array([0.1], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_empty_trajectory(self):
        """Test RMSD calculation with empty trajectory.

        Empty trajectory should return empty array.
        """
        """Test RMSD calculation with empty trajectory.

        Empty trajectory should return empty array.
        """
        empty = md.Trajectory(np.empty((0, 2, 3), dtype=np.float32), self.traj.topology)
        calc = RMSDCalculator([empty], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, None, "mean")
        expected = [np.array([], dtype=np.float32)]
        np.testing.assert_array_equal(result, expected)

    def test_single_atom_trajectory(self):
        """Test RMSD calculation with single-atom trajectory.

        Single atom trajectory should compute proper RMSD values.
        """
        calc = RMSDCalculator([self.traj_1atom], 10, False)
        ref_1atom = md.Trajectory(np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32), self.traj_1atom.topology)
        result = calc.rmsd_to_reference(ref_1atom, None, "mean")
        expected = [np.array([0.0, 0.1], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_window_size_equals_trajectory_length(self):
        """Test window RMSD when window size equals trajectory length.

        Window size equals trajectory should compute single window RMSD.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 5, 1, "mean", "frame_to_start")
        # Single window with all frames to first frame
        expected = [np.array([0.25], dtype=np.float32)]  # mean([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_window_size_exceeds_trajectory(self):
        """Test window RMSD error when window size exceeds trajectory.

        Expected ValueError for window size larger than trajectory length.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="window_size exceeds trajectory length"):
            calc.window(None, 10, 1, "mean", "frame_to_start")

    def test_stride_exceeds_window_size(self):
        """Test window RMSD when stride exceeds window size.

        Large stride should still produce single window result.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.window(None, 2, 5, "mean", "frame_to_start")
        # Only one window possible
        expected = [np.array([0.1], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_negative_window_size_error(self):
        """Test window RMSD error for negative window size.

        Expected ValueError for negative window_size parameter.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="window_size must be positive"):
            calc.window(None, -1, 1, "mean", "frame_to_start")

    def test_negative_stride_error(self):
        """Test window RMSD error for negative stride.

        Expected ValueError for negative stride parameter.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="stride must be positive"):
            calc.window(None, 2, -1, "mean", "frame_to_start")

    def test_zero_window_size_error(self):
        """Test window RMSD error for zero window size.

        Expected ValueError for zero window_size parameter.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="window_size must be positive"):
            calc.window(None, 0, 1, "mean", "frame_to_start")

    def test_zero_stride_error(self):
        """Test window RMSD error for zero stride.

        Expected ValueError for zero stride parameter.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="stride must be positive"):
            calc.window(None, 2, 0, "mean", "frame_to_start")

    def test_atom_indices_out_of_bounds_error(self):
        """Test RMSD error for out-of-bounds atom indices.

        Expected IndexError for atom indices exceeding topology bounds.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(IndexError):
            calc.rmsd_to_reference(self.ref_traj, [10], "mean")

    def test_empty_atom_indices_error(self):
        """Test RMSD calculation with empty atom indices.

        Empty atom indices should result in NaN values.
        Also expects RuntimeWarning from numpy operations on empty arrays.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.warns(RuntimeWarning):
            result = calc.rmsd_to_reference(self.ref_traj, [], "mean")
        # Empty indices results in NaN arrays
        expected = [np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)]
        np.testing.assert_array_equal(result, expected)

    def test_duplicate_atom_indices(self):
        """Test RMSD calculation with duplicate atom indices.

        Duplicate atom indices should be handled gracefully without error.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        result = calc.rmsd_to_reference(self.ref_traj, [0, 0], "mean")
        # Should handle duplicates gracefully
        expected = [np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_invalid_metric_error_to_reference(self):
        """Test rmsd_to_reference error with invalid metric parameter.

        Invalid metric should raise ValueError.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="Unknown metric"):
            calc.rmsd_to_reference(self.ref_traj, None, "invalid")

    def test_invalid_metric_error_frame_to_frame(self):
        """Test frame_to_frame error with invalid metric parameter.

        Invalid metric should raise ValueError.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="Unknown metric"):
            calc.frame_to_frame(None, lag=1, metric="invalid")

    def test_invalid_metric_error_window_frame_to_start(self):
        """Test window frame_to_start error with invalid metric parameter.

        Invalid metric should raise ValueError.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="Unknown metric"):
            calc.window(None, 2, 1, "invalid", "frame_to_start")

    def test_invalid_metric_error_window_frame_to_frame(self):
        """Test window frame_to_frame error with invalid metric parameter.

        Invalid metric should raise ValueError.
        """
        calc = RMSDCalculator([self.traj], 10, False)
        with pytest.raises(ValueError, match="Unknown metric"):
            calc.window(None, 2, 1, "invalid", "frame_to_frame", lag=1)

    def test_window_frame_to_frame_with_chunking(self):
        """Test window frame_to_frame RMSD with chunking enabled.

        Window size=2, stride=1, lag=1, chunk_size=2, use_memmap=True.
        """
        calc = RMSDCalculator([self.traj], chunk_size=2, use_memmap=True)
        result = calc.window(None, 2, 1, "mean", "frame_to_frame", lag=1)
        expected = [np.array([0.1, 0.224, 0.361, 0.5], dtype=np.float32)]
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    # 7. Chunking Verification Tests - Exact Frame Index Testing

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_rmsd_to_reference_chunking_enabled(self, mock_compute):
        """Test rmsd_to_reference with chunking enabled tracks correct frame indices.

        Verifies that _compute_squared_differences is called with correct frame chunks.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called for each chunk: [0,1,2], [3,4,5], [6,7,8], [9]
        assert mock_compute.call_count == 4

        # Extract coords argument from each call (using keyword arguments)
        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Verify each chunk contains exact frame coordinates
        # Chunk 0: frames 0,1,2 with coordinates [0,0,0], [1,1,1], [2,2,2]
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [2., 2., 2.])

        # Chunk 1: frames 3,4,5 with coordinates [3,3,3], [4,4,4], [5,5,5]
        assert chunk_coords[1].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [5., 5., 5.])

        # Chunk 2: frames 6,7,8 with coordinates [6,6,6], [7,7,7], [8,8,8]
        assert chunk_coords[2].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [8., 8., 8.])

        # Chunk 3: frame 9 with coordinates [9,9,9]
        assert chunk_coords[3].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_rmsd_to_reference_chunking_disabled_memmap_false(self, mock_compute):
        """Test rmsd_to_reference with chunking disabled (memmap=False).

        Verifies no chunking occurs when memmap is disabled.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=False)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called once with full trajectory containing all frames 0-9
        assert mock_compute.call_count == 1

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Single chunk: frames 0-9 with coordinates [0,0,0] to [9,9,9]
        assert chunk_coords[0].shape[0] == 10
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][4, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[0][5, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[0][6, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[0][7, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[0][8, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[0][9, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_rmsd_to_reference_chunking_disabled_chunk_zero(self, mock_compute):
        """Test rmsd_to_reference with chunking disabled (chunk_size=0).

        Verifies no chunking occurs when chunk_size is 0.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=0, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called once with full trajectory containing all frames 0-9
        assert mock_compute.call_count == 1

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Single chunk: frames 0-9 with coordinates [0,0,0] to [9,9,9]
        assert chunk_coords[0].shape[0] == 10
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][4, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[0][5, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[0][6, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[0][7, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[0][8, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[0][9, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_rmsd_to_reference_chunking_exact_frame_boundaries(self, mock_compute):
        """Test rmsd_to_reference chunking with exact frame boundaries.

        Verifies correct chunking when frames divide evenly by chunk_size.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(6)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called exactly twice: [0,1,2], [3,4,5]
        assert mock_compute.call_count == 2

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Chunk 0: frames 0,1,2 with coordinates [0,0,0], [1,1,1], [2,2,2]
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [2., 2., 2.])

        # Chunk 1: frames 3,4,5 with coordinates [3,3,3], [4,4,4], [5,5,5]
        assert chunk_coords[1].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [5., 5., 5.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_rmsd_to_reference_chunking_single_frame_trajectory(self, mock_compute):
        """Test rmsd_to_reference chunking with single frame trajectory.

        Verifies chunking logic with trajectory smaller than chunk_size.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(1)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called once with single frame
        assert mock_compute.call_count == 1
        call_kwargs = mock_compute.call_args_list[0].kwargs
        coords = call_kwargs['coords']
        assert coords.shape[0] == 1  # Single frame
        np.testing.assert_array_equal(coords[0, 0], [0., 0., 0.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_rmsd_to_reference_chunking_large_chunk_size(self, mock_compute):
        """Test rmsd_to_reference chunking when chunk_size exceeds trajectory.

        Verifies no chunking when chunk_size is larger than trajectory.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(5)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=10, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should process all frames in one call
        assert mock_compute.call_count == 1

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Single chunk: frames 0-4 with coordinates [0,0,0] to [4,4,4]
        assert chunk_coords[0].shape[0] == 5
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][4, 0], [4., 4., 4.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_rmsd_to_reference_chunking_remainder(self, mock_compute):
        """Test rmsd_to_reference chunking with remainder frames.

        Verifies correct handling of remainder frames that don't fill last chunk.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(7)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called 3 times: [0,1,2], [3,4,5], [6]
        assert mock_compute.call_count == 3

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Chunk 0: frames 0,1,2 with coordinates [0,0,0], [1,1,1], [2,2,2]
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [2., 2., 2.])

        # Chunk 1: frames 3,4,5 with coordinates [3,3,3], [4,4,4], [5,5,5]
        assert chunk_coords[1].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [5., 5., 5.])

        # Chunk 2: frame 6 with coordinates [6,6,6]
        assert chunk_coords[2].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [6., 6., 6.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_frame_to_frame_chunking_enabled(self, mock_compute):
        """Test frame_to_frame with chunking enabled tracks correct frame pairs.

        Verifies that chunking processes correct consecutive frame pairs.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(6)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.frame_to_frame(None, lag=1, metric="mean")

        # Should be called for frame pairs in chunks
        # Frame_to_frame with lag=1 on 6 frames creates 5 pairs: (0,1), (1,2), (2,3), (3,4), (4,5)
        # With chunk_size=3, these should be processed in chunks
        assert mock_compute.call_count == 2  # Exactly 2 chunks

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Chunk 0: target frames 1,2,3 (from pairs (0,1), (1,2), (2,3))
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])

        # Chunk 1: target frames 4,5 (from pairs (3,4), (4,5))
        assert chunk_coords[1].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [5., 5., 5.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_frame_to_frame_chunking_disabled_memmap_false(self, mock_compute):
        """Test frame_to_frame with chunking disabled (memmap=False).

        Verifies no chunking occurs when memmap is disabled.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(6)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=False)

        calc.frame_to_frame(None, lag=1, metric="mean")

        # Should process all frame pairs at once
        assert mock_compute.call_count == 1
        call_kwargs = mock_compute.call_args_list[0].kwargs
        coords = call_kwargs['coords']
        assert coords.shape[0] == 5

        # For frame_to_frame with lag=1, coords contains target frames 1,2,3,4,5
        # paired with reference frames 0,1,2,3,4
        for i in range(5):
            target_frame = i + 1  # Target frames are offset by lag
            np.testing.assert_array_equal(coords[i, 0], [float(target_frame), float(target_frame), float(target_frame)])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_frame_to_frame_chunking_disabled_chunk_zero(self, mock_compute):
        """Test frame_to_frame with chunking disabled (chunk_size=0).

        Verifies no chunking occurs when chunk_size is 0.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(6)
        calc = RMSDCalculator([traj], chunk_size=0, use_memmap=True)

        calc.frame_to_frame(None, lag=1, metric="mean")

        # Should process all frame pairs at once
        assert mock_compute.call_count == 1
        call_kwargs = mock_compute.call_args_list[0].kwargs
        coords = call_kwargs['coords']
        assert coords.shape[0] == 5

        # For frame_to_frame with lag=1, coords contains target frames 1,2,3,4,5
        # paired with reference frames 0,1,2,3,4
        for i in range(5):
            target_frame = i + 1  # Target frames are offset by lag
            np.testing.assert_array_equal(coords[i, 0], [float(target_frame), float(target_frame), float(target_frame)])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_frame_to_frame_chunking_large_chunk_size(self, mock_compute):
        """Test frame_to_frame chunking when chunk_size exceeds trajectory.

        Verifies no chunking when chunk_size is larger than needed.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(4)
        calc = RMSDCalculator([traj], chunk_size=10, use_memmap=True)

        calc.frame_to_frame(None, lag=1, metric="mean")

        # Should process all frame pairs in one call
        assert mock_compute.call_count == 1
        call_kwargs = mock_compute.call_args_list[0].kwargs
        coords = call_kwargs['coords']
        assert coords.shape[0] == 3  # 3 consecutive pairs

        # For frame_to_frame with lag=1, coords contains frames 1,2,3 (targets)
        # paired with frames 0,1,2 (references)
        np.testing.assert_array_equal(coords[0, 0], [1., 1., 1.])  # Frame 1
        np.testing.assert_array_equal(coords[1, 0], [2., 2., 2.])  # Frame 2
        np.testing.assert_array_equal(coords[2, 0], [3., 3., 3.])  # Frame 3

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_frame_to_frame_chunking_high_lag(self, mock_compute):
        """Test frame_to_frame chunking with high lag value.

        Verifies chunking behavior when lag reduces available frame pairs.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(8)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.frame_to_frame(None, lag=3, metric="mean")

        # With lag=3 on 8 frames: pairs (0,3), (1,4), (2,5), (3,6), (4,7) = 5 pairs
        # With chunk_size=3, these should be processed in 2 chunks
        assert mock_compute.call_count == 2

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Chunk 0: target frames 3,4,5 (from pairs (0,3), (1,4), (2,5))
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [5., 5., 5.])

        # Chunk 1: target frames 6,7 (from pairs (3,6), (4,7))
        assert chunk_coords[1].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [7., 7., 7.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_frame_to_frame_chunking_single_pair(self, mock_compute):
        """Test frame_to_frame chunking with minimal trajectory (2 frames).

        Verifies chunking with only one possible frame pair.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(2)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.frame_to_frame(None, lag=1, metric="mean")

        # Should be called once with single pair
        assert mock_compute.call_count == 1
        call_kwargs = mock_compute.call_args_list[0].kwargs
        coords = call_kwargs['coords']
        assert coords.shape[0] == 1  # Single pair

        # For frame_to_frame with lag=1 on 2-frame trajectory, target is frame 1
        np.testing.assert_array_equal(coords[0, 0], [1., 1., 1.])  # Frame 1

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_start_chunking_enabled(self, mock_compute):
        """Test window frame_to_start with chunking enabled.

        Verifies chunking when window_size > chunk_size and memmap=True.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.window(None, window_size=5, stride=1, metric="mean", mode="frame_to_start")

        # 6 windows: [0-4], [1-5], [2-6], [3-7], [4-8], [5-9]
        # Each window: frame 0 is reference, frames 1-4 are chunked as [1-3] and [4]
        # Total: 6 windows × 2 chunks = 12 calls
        assert mock_compute.call_count == 12

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-4]: reference=frame0, compare frames 1,2,3,4
        # Chunk 0: frames 1,2,3
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])
        # Chunk 1: frame 4
        assert chunk_coords[1].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [4., 4., 4.])

        # Window 1 [1-5]: reference=frame1, compare frames 2,3,4,5
        # Chunk 0: frames 2,3,4
        assert chunk_coords[2].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [4., 4., 4.])
        # Chunk 1: frame 5
        assert chunk_coords[3].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [5., 5., 5.])

        # Window 2 [2-6]: reference=frame2, compare frames 3,4,5,6
        # Chunk 0: frames 3,4,5
        assert chunk_coords[4].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[4][2, 0], [5., 5., 5.])
        # Chunk 1: frame 6
        assert chunk_coords[5].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])

        # Window 3 [3-7]: reference=frame3, compare frames 4,5,6,7
        # Chunk 0: frames 4,5,6
        assert chunk_coords[6].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[6][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[6][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[6][2, 0], [6., 6., 6.])
        # Chunk 1: frame 7
        assert chunk_coords[7].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[7][0, 0], [7., 7., 7.])

        # Window 4 [4-8]: reference=frame4, compare frames 5,6,7,8
        # Chunk 0: frames 5,6,7
        assert chunk_coords[8].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[8][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[8][1, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[8][2, 0], [7., 7., 7.])
        # Chunk 1: frame 8
        assert chunk_coords[9].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[9][0, 0], [8., 8., 8.])

        # Window 5 [5-9]: reference=frame5, compare frames 6,7,8,9
        # Chunk 0: frames 6,7,8
        assert chunk_coords[10].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[10][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[10][1, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[10][2, 0], [8., 8., 8.])
        # Chunk 1: frame 9
        assert chunk_coords[11].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[11][0, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_start_chunking_disabled_small_window(self, mock_compute):
        """Test window frame_to_start with chunking disabled (small window).

        Verifies no chunking when window_size <= chunk_size.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=5, use_memmap=True)

        calc.window(None, window_size=3, stride=1, metric="mean", mode="frame_to_start")

        # 8 windows: [0-2], [1-3], [2-4], [3-5], [4-6], [5-7], [6-8], [7-9]
        # No chunking: window_size(3) <= chunk_size(5)
        # Each window: frame 0 is reference, frames 1-2 are compared
        assert mock_compute.call_count == 8

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-2]: reference=frame0, compare frames 1,2
        assert chunk_coords[0].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])

        # Window 1 [1-3]: reference=frame1, compare frames 2,3
        assert chunk_coords[1].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])

        # Window 2 [2-4]: reference=frame2, compare frames 3,4
        assert chunk_coords[2].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])

        # Window 3 [3-5]: reference=frame3, compare frames 4,5
        assert chunk_coords[3].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[3][1, 0], [5., 5., 5.])

        # Window 4 [4-6]: reference=frame4, compare frames 5,6
        assert chunk_coords[4].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [6., 6., 6.])

        # Window 5 [5-7]: reference=frame5, compare frames 6,7
        assert chunk_coords[5].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[5][1, 0], [7., 7., 7.])

        # Window 6 [6-8]: reference=frame6, compare frames 7,8
        assert chunk_coords[6].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[6][0, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[6][1, 0], [8., 8., 8.])

        # Window 7 [7-9]: reference=frame7, compare frames 8,9
        assert chunk_coords[7].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[7][0, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[7][1, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_start_chunking_disabled_memmap_false(self, mock_compute):
        """Test window frame_to_start with chunking disabled (memmap=False).

        Verifies no chunking occurs when memmap is disabled.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=False)

        calc.window(None, window_size=5, stride=1, metric="mean", mode="frame_to_start")

        # 6 windows: [0-4], [1-5], [2-6], [3-7], [4-8], [5-9]
        # No chunking when memmap=False
        # Each window: frame 0 is reference, frames 1-4 are compared
        assert mock_compute.call_count == 6

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-4]: reference=frame0, compare frames 1,2,3,4
        assert chunk_coords[0].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [4., 4., 4.])

        # Window 1 [1-5]: reference=frame1, compare frames 2,3,4,5
        assert chunk_coords[1].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][3, 0], [5., 5., 5.])

        # Window 2 [2-6]: reference=frame2, compare frames 3,4,5,6
        assert chunk_coords[2].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[2][3, 0], [6., 6., 6.])

        # Window 3 [3-7]: reference=frame3, compare frames 4,5,6,7
        assert chunk_coords[3].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[3][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[3][2, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[3][3, 0], [7., 7., 7.])

        # Window 4 [4-8]: reference=frame4, compare frames 5,6,7,8
        assert chunk_coords[4].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[4][2, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[4][3, 0], [8., 8., 8.])

        # Window 5 [5-9]: reference=frame5, compare frames 6,7,8,9
        assert chunk_coords[5].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[5][1, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[5][2, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[5][3, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_start_chunking_disabled_chunk_zero(self, mock_compute):
        """Test window frame_to_start with chunking disabled (chunk_size=0).

        Verifies no chunking occurs when chunk_size is 0.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=0, use_memmap=True)

        calc.window(None, window_size=5, stride=1, metric="mean", mode="frame_to_start")

        # 6 windows: [0-4], [1-5], [2-6], [3-7], [4-8], [5-9]
        # No chunking when chunk_size=0
        # Each window: frame 0 is reference, frames 1-4 are compared
        assert mock_compute.call_count == 6

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-4]: reference=frame0, compare frames 1,2,3,4
        assert chunk_coords[0].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [4., 4., 4.])

        # Window 1 [1-5]: reference=frame1, compare frames 2,3,4,5
        assert chunk_coords[1].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][3, 0], [5., 5., 5.])

        # Window 2 [2-6]: reference=frame2, compare frames 3,4,5,6
        assert chunk_coords[2].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[2][3, 0], [6., 6., 6.])

        # Window 3 [3-7]: reference=frame3, compare frames 4,5,6,7
        assert chunk_coords[3].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[3][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[3][2, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[3][3, 0], [7., 7., 7.])

        # Window 4 [4-8]: reference=frame4, compare frames 5,6,7,8
        assert chunk_coords[4].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[4][2, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[4][3, 0], [8., 8., 8.])

        # Window 5 [5-9]: reference=frame5, compare frames 6,7,8,9
        assert chunk_coords[5].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[5][1, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[5][2, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[5][3, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_start_chunking_boundary_window_equals_chunk(self, mock_compute):
        """Test window frame_to_start boundary case (window_size == chunk_size).

        Verifies no chunking when window_size equals chunk_size exactly.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=4, use_memmap=True)

        calc.window(None, window_size=4, stride=1, metric="mean", mode="frame_to_start")

        # 7 windows: [0-3], [1-4], [2-5], [3-6], [4-7], [5-8], [6-9]
        # No chunking: window_size(4) == chunk_size(4)
        # Each window: frame 0 is reference, frames 1-3 are compared
        assert mock_compute.call_count == 7

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-3]: reference=frame0, compare frames 1,2,3
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])

        # Window 1 [1-4]: reference=frame1, compare frames 2,3,4
        assert chunk_coords[1].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [4., 4., 4.])

        # Window 2 [2-5]: reference=frame2, compare frames 3,4,5
        assert chunk_coords[2].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [5., 5., 5.])

        # Window 3 [3-6]: reference=frame3, compare frames 4,5,6
        assert chunk_coords[3].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[3][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[3][2, 0], [6., 6., 6.])

        # Window 4 [4-7]: reference=frame4, compare frames 5,6,7
        assert chunk_coords[4].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[4][2, 0], [7., 7., 7.])

        # Window 5 [5-8]: reference=frame5, compare frames 6,7,8
        assert chunk_coords[5].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[5][1, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[5][2, 0], [8., 8., 8.])

        # Window 6 [6-9]: reference=frame6, compare frames 7,8,9
        assert chunk_coords[6].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[6][0, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[6][1, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[6][2, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_start_chunking_large_stride(self, mock_compute):
        """Test window frame_to_start chunking with large stride.

        Verifies chunking behavior when stride creates fewer windows.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.window(None, window_size=5, stride=3, metric="mean", mode="frame_to_start")

        # 2 windows with stride=3: [0-4], [3-7] (starts at 0, 3)
        # Each window: frame 0 is reference, frames 1-4 chunked as [1-3] and [4]
        # Total: 2 windows × 2 chunks = 4 calls
        assert mock_compute.call_count == 4

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-4]: reference=frame0, compare frames 1,2,3,4
        # Chunk 0: frames 1,2,3
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])
        # Chunk 1: frame 4
        assert chunk_coords[1].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [4., 4., 4.])

        # Window 1 [3-7]: reference=frame3, compare frames 4,5,6,7
        # Chunk 0: frames 4,5,6
        assert chunk_coords[2].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [6., 6., 6.])
        # Chunk 1: frame 7
        assert chunk_coords[3].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [7., 7., 7.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_start_chunking_single_window(self, mock_compute):
        """Test window frame_to_start chunking with single large window.

        Verifies chunking when only one window spans entire trajectory.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(8)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.window(None, window_size=8, stride=10, metric="mean", mode="frame_to_start")

        # 1 window: [0-7] (stride=10 > available range, so only one window)
        # Window: reference=frame0, compare frames 1,2,3,4,5,6,7
        # Frames 1-7 chunked by 3: [1-3], [4-6], [7] = 3 chunks
        assert mock_compute.call_count == 3

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Chunk 0: frames 1,2,3
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])

        # Chunk 1: frames 4,5,6
        assert chunk_coords[1].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [6., 6., 6.])

        # Chunk 2: frame 7
        assert chunk_coords[2].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [7., 7., 7.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_frame_chunking_enabled(self, mock_compute):
        """Test window frame_to_frame with chunking enabled.

        Verifies chunking when window_size > chunk_size and memmap=True.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.window(None, window_size=5, stride=1, metric="mean", mode="frame_to_frame", lag=1)

        # 6 windows: [0-4], [1-5], [2-6], [3-7], [4-8], [5-9]
        # Each window has 4 pairs with lag=1: total 4 pairs per window
        # 4 pairs chunked by 3: 2 chunks per window (3+1)
        # Total: 6 windows × 2 chunks = 12 calls
        assert mock_compute.call_count == 12

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-4]: pairs (0,1),(1,2),(2,3),(3,4)
        # Chunk 0: target frames 1,2,3
        assert chunk_coords[0].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])
        # Chunk 1: target frame 4
        assert chunk_coords[1].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [4., 4., 4.])

        # Window 1 [1-5]: pairs (1,2),(2,3),(3,4),(4,5)
        # Chunk 0: target frames 2,3,4
        assert chunk_coords[2].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [4., 4., 4.])
        # Chunk 1: target frame 5
        assert chunk_coords[3].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [5., 5., 5.])

        # Window 2 [2-6]: pairs (2,3),(3,4),(4,5),(5,6)
        # Chunk 0: target frames 3,4,5
        assert chunk_coords[4].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[4][2, 0], [5., 5., 5.])
        # Chunk 1: target frame 6
        assert chunk_coords[5].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])

        # Window 3 [3-7]: pairs (3,4),(4,5),(5,6),(6,7)
        # Chunk 0: target frames 4,5,6
        assert chunk_coords[6].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[6][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[6][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[6][2, 0], [6., 6., 6.])
        # Chunk 1: target frame 7
        assert chunk_coords[7].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[7][0, 0], [7., 7., 7.])

        # Window 4 [4-8]: pairs (4,5),(5,6),(6,7),(7,8)
        # Chunk 0: target frames 5,6,7
        assert chunk_coords[8].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[8][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[8][1, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[8][2, 0], [7., 7., 7.])
        # Chunk 1: target frame 8
        assert chunk_coords[9].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[9][0, 0], [8., 8., 8.])

        # Window 5 [5-9]: pairs (5,6),(6,7),(7,8),(8,9)
        # Chunk 0: target frames 6,7,8
        assert chunk_coords[10].shape[0] == 3
        np.testing.assert_array_equal(chunk_coords[10][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[10][1, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[10][2, 0], [8., 8., 8.])
        # Chunk 1: target frame 9
        assert chunk_coords[11].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[11][0, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_frame_chunking_disabled_small_window(self, mock_compute):
        """Test window frame_to_frame with chunking disabled (small window).

        Verifies no chunking when window_size <= chunk_size.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=5, use_memmap=True)

        calc.window(None, window_size=3, stride=1, metric="mean", mode="frame_to_frame", lag=1)

        # 8 windows: [0-2], [1-3], [2-4], [3-5], [4-6], [5-7], [6-8], [7-9]
        # Each window has 2 pairs with lag=1
        # No chunking: window_size(3) <= chunk_size(5)
        assert mock_compute.call_count == 8

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-2]: pairs (0,1),(1,2) → target frames 1,2
        assert chunk_coords[0].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])

        # Window 1 [1-3]: pairs (1,2),(2,3) → target frames 2,3
        assert chunk_coords[1].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])

        # Window 2 [2-4]: pairs (2,3),(3,4) → target frames 3,4
        assert chunk_coords[2].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])

        # Window 3 [3-5]: pairs (3,4),(4,5) → target frames 4,5
        assert chunk_coords[3].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[3][1, 0], [5., 5., 5.])

        # Window 4 [4-6]: pairs (4,5),(5,6) → target frames 5,6
        assert chunk_coords[4].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [6., 6., 6.])

        # Window 5 [5-7]: pairs (5,6),(6,7) → target frames 6,7
        assert chunk_coords[5].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[5][1, 0], [7., 7., 7.])

        # Window 6 [6-8]: pairs (6,7),(7,8) → target frames 7,8
        assert chunk_coords[6].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[6][0, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[6][1, 0], [8., 8., 8.])

        # Window 7 [7-9]: pairs (7,8),(8,9) → target frames 8,9
        assert chunk_coords[7].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[7][0, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[7][1, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_frame_chunking_disabled_memmap_false(self, mock_compute):
        """Test window frame_to_frame with chunking disabled (memmap=False).

        Verifies no chunking occurs when memmap is disabled.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=False)

        calc.window(None, window_size=5, stride=1, metric="mean", mode="frame_to_frame", lag=1)

        # 6 windows: [0-4], [1-5], [2-6], [3-7], [4-8], [5-9]
        # Each window has 4 pairs with lag=1
        # No chunking when memmap=False
        assert mock_compute.call_count == 6

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-4]: pairs (0,1),(1,2),(2,3),(3,4) → target frames 1,2,3,4
        assert chunk_coords[0].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [4., 4., 4.])

        # Window 1 [1-5]: pairs (1,2),(2,3),(3,4),(4,5) → target frames 2,3,4,5
        assert chunk_coords[1].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][3, 0], [5., 5., 5.])

        # Window 2 [2-6]: pairs (2,3),(3,4),(4,5),(5,6) → target frames 3,4,5,6
        assert chunk_coords[2].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[2][3, 0], [6., 6., 6.])

        # Window 3 [3-7]: pairs (3,4),(4,5),(5,6),(6,7) → target frames 4,5,6,7
        assert chunk_coords[3].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[3][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[3][2, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[3][3, 0], [7., 7., 7.])

        # Window 4 [4-8]: pairs (4,5),(5,6),(6,7),(7,8) → target frames 5,6,7,8
        assert chunk_coords[4].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[4][2, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[4][3, 0], [8., 8., 8.])

        # Window 5 [5-9]: pairs (5,6),(6,7),(7,8),(8,9) → target frames 6,7,8,9
        assert chunk_coords[5].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[5][1, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[5][2, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[5][3, 0], [9., 9., 9.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_window_frame_to_frame_chunking_disabled_chunk_zero(self, mock_compute):
        """Test window frame_to_frame with chunking disabled (chunk_size=0).

        Verifies no chunking occurs when chunk_size is 0.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(10)
        calc = RMSDCalculator([traj], chunk_size=0, use_memmap=True)

        calc.window(None, window_size=5, stride=1, metric="mean", mode="frame_to_frame", lag=1)

        # 6 windows: [0-4], [1-5], [2-6], [3-7], [4-8], [5-9]
        # Each window has 4 pairs with lag=1
        # No chunking when chunk_size=0
        assert mock_compute.call_count == 6

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-4]: pairs (0,1),(1,2),(2,3),(3,4) → target frames 1,2,3,4
        assert chunk_coords[0].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [4., 4., 4.])

        # Window 1 [1-5]: pairs (1,2),(2,3),(3,4),(4,5) → target frames 2,3,4,5
        assert chunk_coords[1].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[1][2, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[1][3, 0], [5., 5., 5.])

        # Window 2 [2-6]: pairs (2,3),(3,4),(4,5),(5,6) → target frames 3,4,5,6
        assert chunk_coords[2].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[2][2, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[2][3, 0], [6., 6., 6.])

        # Window 3 [3-7]: pairs (3,4),(4,5),(5,6),(6,7) → target frames 4,5,6,7
        assert chunk_coords[3].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [4., 4., 4.])
        np.testing.assert_array_equal(chunk_coords[3][1, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[3][2, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[3][3, 0], [7., 7., 7.])

        # Window 4 [4-8]: pairs (4,5),(5,6),(6,7),(7,8) → target frames 5,6,7,8
        assert chunk_coords[4].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [5., 5., 5.])
        np.testing.assert_array_equal(chunk_coords[4][1, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[4][2, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[4][3, 0], [8., 8., 8.])

        # Window 5 [5-9]: pairs (5,6),(6,7),(7,8),(8,9) → target frames 6,7,8,9
        assert chunk_coords[5].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [6., 6., 6.])
        np.testing.assert_array_equal(chunk_coords[5][1, 0], [7., 7., 7.])
        np.testing.assert_array_equal(chunk_coords[5][2, 0], [8., 8., 8.])
        np.testing.assert_array_equal(chunk_coords[5][3, 0], [9., 9., 9.])

    def test_chunking_edge_case_empty_trajectory(self):
        """Test chunking behavior with empty trajectory.

        Verifies graceful handling of edge case with no frames.
        """
        # Create empty trajectory by slicing existing trajectory to 0 frames
        empty_traj = self.traj[:0]  # This creates trajectory with 0 frames but same topology
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([empty_traj], chunk_size=3, use_memmap=True)

        result = calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should handle empty trajectory gracefully
        assert len(result[0]) == 0

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_rmsd_to_reference_chunk_size_one(self, mock_compute):
        """Test rmsd_to_reference chunking behavior with chunk_size=1.

        Verifies correct frame-by-frame processing.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(5)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=1, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called 5 times, once per frame
        assert mock_compute.call_count == 5

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Chunk 0: frame 0 with coordinates [0,0,0]
        assert chunk_coords[0].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])

        # Chunk 1: frame 1 with coordinates [1,1,1]
        assert chunk_coords[1].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [1., 1., 1.])

        # Chunk 2: frame 2 with coordinates [2,2,2]
        assert chunk_coords[2].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [2., 2., 2.])

        # Chunk 3: frame 3 with coordinates [3,3,3]
        assert chunk_coords[3].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [3., 3., 3.])

        # Chunk 4: frame 4 with coordinates [4,4,4]
        assert chunk_coords[4].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [4., 4., 4.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_frame_to_frame_chunk_size_one(self, mock_compute):
        """Test frame_to_frame chunking behavior with chunk_size=1.

        Verifies correct pair-by-pair processing.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(4)
        calc = RMSDCalculator([traj], chunk_size=1, use_memmap=True)

        calc.frame_to_frame(None, lag=1, metric="mean")

        # Should be called 3 times, once per pair: (0,1), (1,2), (2,3)
        assert mock_compute.call_count == 3

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Chunk 0: target frame 1 from pair (0,1)
        assert chunk_coords[0].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])

        # Chunk 1: target frame 2 from pair (1,2)
        assert chunk_coords[1].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])

        # Chunk 2: target frame 3 from pair (2,3)
        assert chunk_coords[2].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_window_frame_to_start_chunk_size_one(self, mock_compute):
        """Test window_frame_to_start chunking behavior with chunk_size=1.

        Verifies correct frame-by-frame processing within windows.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(5)
        calc = RMSDCalculator([traj], chunk_size=1, use_memmap=True)

        calc.window(None, window_size=3, stride=1, metric="mean", mode="frame_to_start")

        # 3 windows: [0-2], [1-3], [2-4]
        # Each window: frame 0 is reference, frames 1-2 are chunked frame-by-frame
        # Total: 3 windows × 2 frames = 6 calls
        assert mock_compute.call_count == 6

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-2]: reference=frame0, compare frames 1,2
        # Chunk 0: frame 1
        assert chunk_coords[0].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        # Chunk 1: frame 2
        assert chunk_coords[1].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])

        # Window 1 [1-3]: reference=frame1, compare frames 2,3
        # Chunk 0: frame 2
        assert chunk_coords[2].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [2., 2., 2.])
        # Chunk 1: frame 3
        assert chunk_coords[3].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [3., 3., 3.])

        # Window 2 [2-4]: reference=frame2, compare frames 3,4
        # Chunk 0: frame 3
        assert chunk_coords[4].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [3., 3., 3.])
        # Chunk 1: frame 4
        assert chunk_coords[5].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [4., 4., 4.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_window_frame_to_frame_chunk_size_one(self, mock_compute):
        """Test window_frame_to_frame chunking behavior with chunk_size=1.

        Verifies correct pair-by-pair processing within windows.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(5)
        calc = RMSDCalculator([traj], chunk_size=1, use_memmap=True)

        calc.window(None, window_size=3, stride=1, metric="mean", mode="frame_to_frame", lag=1)

        # 3 windows: [0-2], [1-3], [2-4]
        # Each window has 2 pairs with lag=1
        # Each pair in separate chunk
        # Total: 3 windows × 2 pairs = 6 calls
        assert mock_compute.call_count == 6

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-2]: pairs (0,1),(1,2) → target frames 1,2
        # Chunk 0: target frame 1
        assert chunk_coords[0].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        # Chunk 1: target frame 2
        assert chunk_coords[1].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])

        # Window 1 [1-3]: pairs (1,2),(2,3) → target frames 2,3
        # Chunk 0: target frame 2
        assert chunk_coords[2].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [2., 2., 2.])
        # Chunk 1: target frame 3
        assert chunk_coords[3].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[3][0, 0], [3., 3., 3.])

        # Window 2 [2-4]: pairs (2,3),(3,4) → target frames 3,4
        # Chunk 0: target frame 3
        assert chunk_coords[4].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[4][0, 0], [3., 3., 3.])
        # Chunk 1: target frame 4
        assert chunk_coords[5].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[5][0, 0], [4., 4., 4.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_rmsd_to_reference_very_large_chunk_size(self, mock_compute):
        """Test rmsd_to_reference chunking behavior with very large chunk_size.

        Verifies no chunking occurs when chunk_size exceeds any realistic needs.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(5)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=1000000, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called once with all frames
        assert mock_compute.call_count == 1

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Single chunk: frames 0-4 with coordinates [0,0,0] to [4,4,4]
        assert chunk_coords[0].shape[0] == 5
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][4, 0], [4., 4., 4.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_frame_to_frame_very_large_chunk_size(self, mock_compute):
        """Test frame_to_frame chunking behavior with very large chunk_size.

        Verifies no chunking occurs when chunk_size exceeds any realistic needs.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(5)
        calc = RMSDCalculator([traj], chunk_size=1000000, use_memmap=True)

        calc.frame_to_frame(None, lag=1, metric="mean")

        # Should be called once with all pairs
        assert mock_compute.call_count == 1

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Single chunk: target frames 1,2,3,4 from pairs (0,1),(1,2),(2,3),(3,4)
        assert chunk_coords[0].shape[0] == 4
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[0][2, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[0][3, 0], [4., 4., 4.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_window_frame_to_start_very_large_chunk_size(self, mock_compute):
        """Test window_frame_to_start chunking behavior with very large chunk_size.

        Verifies no chunking occurs when chunk_size exceeds any realistic needs.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(5)
        calc = RMSDCalculator([traj], chunk_size=1000000, use_memmap=True)

        calc.window(None, window_size=3, stride=1, metric="mean", mode="frame_to_start")

        # 3 windows: [0-2], [1-3], [2-4]
        # No chunking with very large chunk_size
        # Each window: frame 0 is reference, frames 1-2 are compared
        assert mock_compute.call_count == 3

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-2]: reference=frame0, compare frames 1,2
        assert chunk_coords[0].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])

        # Window 1 [1-3]: reference=frame1, compare frames 2,3
        assert chunk_coords[1].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])

        # Window 2 [2-4]: reference=frame2, compare frames 3,4
        assert chunk_coords[2].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_window_frame_to_frame_very_large_chunk_size(self, mock_compute):
        """Test window_frame_to_frame chunking behavior with very large chunk_size.

        Verifies no chunking occurs when chunk_size exceeds any realistic needs.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(5)
        calc = RMSDCalculator([traj], chunk_size=1000000, use_memmap=True)

        calc.window(None, window_size=3, stride=1, metric="mean", mode="frame_to_frame", lag=1)

        # 3 windows: [0-2], [1-3], [2-4]
        # Each window has 2 pairs with lag=1
        # No chunking with very large chunk_size
        assert mock_compute.call_count == 3

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Window 0 [0-2]: pairs (0,1),(1,2) → target frames 1,2
        assert chunk_coords[0].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])
        np.testing.assert_array_equal(chunk_coords[0][1, 0], [2., 2., 2.])

        # Window 1 [1-3]: pairs (1,2),(2,3) → target frames 2,3
        assert chunk_coords[1].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[1][0, 0], [2., 2., 2.])
        np.testing.assert_array_equal(chunk_coords[1][1, 0], [3., 3., 3.])

        # Window 2 [2-4]: pairs (2,3),(3,4) → target frames 3,4
        assert chunk_coords[2].shape[0] == 2
        np.testing.assert_array_equal(chunk_coords[2][0, 0], [3., 3., 3.])
        np.testing.assert_array_equal(chunk_coords[2][1, 0], [4., 4., 4.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_rmsd_to_reference_single_frame(self, mock_compute):
        """Test rmsd_to_reference chunking behavior with single frame trajectory.

        Verifies handling of minimal trajectory.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(1)
        ref_traj = self.make_unique_trajectory(1)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.rmsd_to_reference(ref_traj, None, "mean")

        # Should be called once with single frame
        assert mock_compute.call_count == 1

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Single chunk: frame 0 with coordinates [0,0,0]
        assert chunk_coords[0].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [0., 0., 0.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_frame_to_frame_two_frames(self, mock_compute):
        """Test frame_to_frame chunking behavior with minimal two frame trajectory.

        Verifies handling of minimal trajectory for frame_to_frame.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(2)
        calc = RMSDCalculator([traj], chunk_size=3, use_memmap=True)

        calc.frame_to_frame(None, lag=1, metric="mean")

        # Should be called once with single pair
        assert mock_compute.call_count == 1

        call_kwargs_list = [call.kwargs for call in mock_compute.call_args_list]
        chunk_coords = [kwargs['coords'] for kwargs in call_kwargs_list]

        # Single chunk: target frame 1 from pair (0,1)
        assert chunk_coords[0].shape[0] == 1
        np.testing.assert_array_equal(chunk_coords[0][0, 0], [1., 1., 1.])

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_window_frame_to_start_larger_than_trajectory(self, mock_compute):
        """Test window_frame_to_start chunking behavior when window_size exceeds trajectory length.

        Verifies graceful handling when window cannot be formed.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(3)
        calc = RMSDCalculator([traj], chunk_size=2, use_memmap=True)

        with pytest.raises(ValueError, match="window_size exceeds trajectory length"):
            calc.window(None, window_size=5, stride=1, metric="mean", mode="frame_to_start")

    @patch('mdxplain.analysis.structure.calculators.rmsd_calculator.RMSDCalculator._compute_squared_differences')
    def test_chunking_edge_case_window_frame_to_frame_larger_than_trajectory(self, mock_compute):
        """Test window_frame_to_frame chunking behavior when window_size exceeds trajectory length.

        Verifies graceful handling when window cannot be formed.
        """
        mock_compute.return_value = np.array([[0.0]], dtype=np.float32)

        traj = self.make_unique_trajectory(3)
        calc = RMSDCalculator([traj], chunk_size=2, use_memmap=True)

        with pytest.raises(ValueError, match="window_size exceeds trajectory length"):
            calc.window(None, window_size=5, stride=1, metric="mean", mode="frame_to_frame", lag=1)
